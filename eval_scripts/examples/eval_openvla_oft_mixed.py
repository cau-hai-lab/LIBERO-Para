"""OpenVLA-OFT (Mixed) — thin wrapper around eval_openvla_oft.py"""
import sys
sys.argv += ["--pretrained_checkpoint", "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"]
if "--output_dir" not in sys.argv:
    sys.argv += ["--output_dir", "./logs_para/openvla-oft-mixed/"]
from eval_openvla_oft import main
main()
