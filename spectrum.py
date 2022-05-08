import argparse
import torch

from kernel_functions import dcntk
from utils import hypersphere_random_sampler, kernel_spectrum

torch.set_default_dtype(torch.float64)


parser = argparse.ArgumentParser()

parser.add_argument("--n_points", type=int, help="number of points", default=8192)
parser.add_argument("--imagesize", type=int, help="size of the images", required=True)
parser.add_argument(
    "--filtersizes", nargs="+", type=int, help="size of filters", required=True
)
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
parser.add_argument("--device_diag", choices=["cpu", "auto"], default="auto")

args = parser.parse_args()

print(args, flush=True)


if args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x_list = []
for patch in range(args.imagesize // args.filtersizes[0]):
    x_list.append(hypersphere_random_sampler(args.n_points, args.filtersizes[0]))
x = torch.cat(x_list, dim=1).to(device)

gram = dcntk(x, x, filtersizes=args.filtersizes, normalize=args.normalize)

if args.device_diag == "cpu":
    device_diag = torch.device("cpu")
else:
    device_diag = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evals = kernel_spectrum(gram.to(device_diag))


"""
Log
"""

fsz_list = map(str, args.filtersizes)
fsz_list = "-".join(fsz_list)

filename = f"dcntk_evals_d{args.imagesize}_s{fsz_list}_n{args.n_points}.pt"

torch.save({"args": args, "evals": evals,}, filename)
