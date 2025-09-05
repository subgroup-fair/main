# fairbench/methods/__init__.py
from .dr import run_dr
from .gerryfair_wrapper import run_gerryfair
from .multicalib_wrapper import run_multicalib
from .seq_wrapper import run_sequential
from .reduction_wrapper import run_reduction

def run_method(args, data):
    if args.method == "dr":
        return run_dr(args, data)
    if args.method == "gerryfair":
        return run_gerryfair(args, data)
    if args.method == "multicalib":
        return run_multicalib(args, data)
    if args.method == "sequential":
        return run_sequential(args, data)
    if args.method == "reduction":
        return run_reduction(args, data)
    raise ValueError(args.method)
