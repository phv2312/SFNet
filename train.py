import os
import argparse
import random
import numpy as np
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lrs

from geek_dataset import RandomAugmentPairAnimeDataset
from custom_loss import loss_function
from model import SFNet


global_seed = 0


def parse_args():
    parser = argparse.ArgumentParser(description="SFNet")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="mini-batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs for training")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.2, help="decaying factor")
    parser.add_argument("--decay_schedule", type=str, default="30", help="learning rate decaying schedule")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for data loader")
    parser.add_argument("--feature_h", type=int, default=48, help="height of feature volume")
    parser.add_argument("--feature_w", type=int, default=32, help="width of feature volume")
    parser.add_argument("--train_image_path", type=str, default="./data/training_data/VOC2012_seg_img.npy",
                        help="directory of pre-processed(.npy) images")
    parser.add_argument("--train_mask_path", type=str, default="./data/training_data/VOC2012_seg_msk.npy",
                        help="directory of pre-processed(.npy) foreground masks")
    parser.add_argument("--valid_csv_path", type=str, default="./data/PF_Pascal/bbox_val_pairs_pf_pascal.csv",
                        help="directory of validation csv file")
    parser.add_argument("--valid_image_path", type=str, default="./data/PF_Pascal/",
                        help="directory of validation data")
    parser.add_argument("--beta", type=float, default=50, help="inverse temperature of softmax @ kernel soft argmax")
    parser.add_argument("--kernel_sigma", type=float, default=8,
                        help="standard deviation of Gaussian kerenl @ kernel soft argmax")
    parser.add_argument("--lambda1", type=float, default=3, help="weight parameter of mask consistency loss")
    parser.add_argument("--lambda2", type=float, default=16, help="weight parameter of flow consistency loss")
    parser.add_argument("--lambda3", type=float, default=0.5, help="weight parameter of smoothness loss")
    parser.add_argument("--eval_type", type=str, default="bounding_box", choices=("bounding_box", "image_size"),
                        help="evaluation type for PCK threshold (bounding box | image size)")

    args = parser.parse_args()
    return args


def set_seed(args):
    if args.seed is None:
        args.seed = np.random.randint(10000)
        print("Seed number: ", args.seed)

    global global_seed
    global_seed = args.seed
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def log(text, logger_file):
    # Make a log file & directory for saving weights
    with open(logger_file, "a") as f:
        f.write(text)
        f.close()
    return


def main(args):
    logger_file = "./training_log.txt"
    weight_path = "./weights/best_checkpoint.pt"
    if os.path.exists(logger_file):
        os.remove(logger_file)

    if not os.path.exists("./weights/"):
        os.mkdir("./weights/")

    # Data Loader
    root_dir = "/home/tyler/work/data/GeekInt/data_dc"
    h = 768
    w = 512
    image_size = (w, h)

    train_dataset = RandomAugmentPairAnimeDataset(root_dir, image_size, args.feature_h, args.feature_w)
    print("Dataset size:", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=_init_fn)

    valid_dataset = train_dataset
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers)

    # Instantiate model
    net = SFNet(args.feature_h, args.feature_w, beta=args.beta, kernel_sigma=args.kernel_sigma)

    # Load pre-trained weight
    if os.path.exists(weight_path):
        try:
            best_weights = torch.load(weight_path, map_location="cpu")
            adap3_dict = best_weights["state_dict1"]
            adap4_dict = best_weights["state_dict2"]
            net.adap_layer_feat3.load_state_dict(adap3_dict, strict=False)
            net.adap_layer_feat4.load_state_dict(adap4_dict, strict=False)
        except Exception as e:
            print("exception while loading weight from pre-trained, detail:", str(e))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Instantiate loss
    criterion = loss_function(args).to(device)

    # Instantiate optimizer
    param = list(net.adap_layer_feat3.parameters()) + list(net.adap_layer_feat4.parameters())
    optimizer = torch.optim.Adam(param, lr=args.lr)
    decay_schedule = list(map(lambda x: int(x), args.decay_schedule.split("-")))
    scheduler = lrs.MultiStepLR(optimizer, milestones=decay_schedule, gamma=args.gamma)

    # PCK metric from "https://github.com/ignacio-rocco/weakalign/blob/master/util/eval_util.py"
    def correct_keypoints(source_points, warped_points, l_pck, alpha=0.1):
        # compute correct keypoints
        p_src = source_points[0, :]
        p_wrp = warped_points[0, :]

        n_pts = torch.sum(torch.ne(p_src[0, :], -1) * torch.ne(p_src[1, :], -1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:, :n_pts] - p_wrp[:, :n_pts], 2), 0), 0.5)
        l_pck_mat = l_pck[0].expand_as(point_distance)
        correct_points = torch.le(point_distance, l_pck_mat * alpha)
        pck = torch.mean(correct_points.float())
        return pck

    # Training
    best_pck = 0
    print("Training started")
    for ep in range(args.epochs):
        print("Current epoch : %d" % ep)
        log("Current epoch : %d\n" % ep, logger_file)
        log("Current learning rate : %e\n" % optimizer.state_dict()["param_groups"][0]["lr"], logger_file)

        net.train()
        net.feature_extraction.eval()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            src_image = batch["image1"].to(device)
            tgt_image = batch["image2"].to(device)
            gt_src_mask = batch["mask1"].to(device)
            gt_tgt_mask = batch["mask2"].to(device)

            output = net(src_image, tgt_image, gt_src_mask, gt_tgt_mask)

            optimizer.zero_grad()
            loss, l1, l2, l3 = criterion(output, gt_src_mask, gt_tgt_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print("Epoch %03d (%04d/%04d) = Loss : %5f (Now : %5f)\t" % (
                ep, i, len(train_dataset) // args.batch_size, total_loss / (i + 1), loss.cpu().data))
            print("L1 : %5f, L2 : %5f, L3 : %5f\n" % (l1.item(), l2.item(), l3.item()))

        scheduler.step()
        print("Epoch %03d finished... Average loss : %5f\n" % (ep, total_loss / len(train_loader)))

        torch.save({
            "state_dict1": net.adap_layer_feat3.state_dict(),
            "state_dict2": net.adap_layer_feat4.state_dict()
        }, weight_path)

    a = torch.from_numpy
    print("Done")


if __name__ == "__main__":
    main(parse_args())
