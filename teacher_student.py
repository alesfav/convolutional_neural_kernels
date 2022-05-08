import argparse
import torch

from kernel_functions import dcntk
from utils import hypersphere_random_sampler, grf_generator, kernel_regression

torch.set_default_dtype(torch.float64)


parser = argparse.ArgumentParser()

parser.add_argument("--testsize", type=int, help="size of the test set", default=8192)
parser.add_argument("--imagesize", type=int, help="size of the images", required=True)
parser.add_argument(
    "--patternsizes", nargs="+", type=int, help="sizes of teach. filters", required=True
)
parser.add_argument(
    "--filtersizes", nargs="+", type=int, help="size of student filters", required=True
)
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--ridge", type=float, help="regularisation", default=0.0)
parser.add_argument("--exp", type=int, help="index for experiment", default=0)
parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")

args = parser.parse_args()

print(args, flush=True)


if args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


trainsizes = [128, 256, 512, 1024, 2048, 4096, 8192]


for trainsize in trainsizes:

    print(f"Starting P = {trainsize}...", flush=True)

    """
    Teacher
    """

    x_list = []
    for patch in range(args.imagesize // args.filtersizes[0]):
        x_list.append(
            hypersphere_random_sampler(trainsize + args.testsize, args.filtersizes[0])
        )
    x = torch.cat(x_list, dim=1).to(device)

    teacher_covariance = dcntk(
        x, x, filtersizes=args.patternsizes, normalize=args.normalize
    )
    y = grf_generator(teacher_covariance, device)

    x_train = x[:trainsize]
    y_train = y[:trainsize]

    x_test = x[trainsize:]
    y_test = y[trainsize:]

    """
    Student
    """

    student_trtr = dcntk(
        x_train, x_train, filtersizes=args.filtersizes, normalize=args.normalize
    )
    student_tetr = dcntk(
        x_test, x_train, filtersizes=args.filtersizes, normalize=args.normalize
    )

    mse = kernel_regression(
        student_trtr, student_tetr, y_train, y_test, args.ridge, device
    )

    """
    Log
    """

    psz_list = map(str, args.patternsizes)
    psz_list = "-".join(psz_list)

    fsz_list = map(str, args.filtersizes)
    fsz_list = "-".join(fsz_list)

    filename = (
        f"dcntk_d{args.imagesize}_t{psz_list}_s{fsz_list}_"
        f"n{trainsize}_r{args.ridge}_exp{args.exp}.pt"
    )

    torch.save({"args": args, "mse": mse,}, filename)
