import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pathlib import Path

# Optional: clear proxy variables here if you do not want PyCharm / VPN proxies to affect requests.
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

from hitom_llm.main import run_pipeline

# PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "conditions/0_forward_belief_true_belief/stories.csv"
OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_bigtom_true_belief_token_5.4"
# OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_bigtom_true_belief_gemini_3"
# DATA_PATH = PROJECT_ROOT / "Hi-ToM_data.json"
# OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_hitom_token_gemini"
# DATA_PATH = PROJECT_ROOT / "fantom.json"
# OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_fantom_token_5.4"
# # OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_hitom_CoT"
# # OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_hitom_gemini_3"
# # OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_fantom_5.4_CoT"
# # OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_hitom_gemini_3_CoT"
# # OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_fantom_gemini_3_CoT"
# # OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_fantom_gemini_3"
# OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_hitom_5.4_CoT_again"
PROVIDER = "openai"
# MODEL = "gpt-5.4-nano"
MODEL = "gpt-5.4"
# PROVIDER = "gemini"
# MODEL = "gemini-3-flash-preview"
SAMPLE_LIMIT = 50
DEBUG = True
DATASET_TYPE = "bigtom"
# DATASET_TYPE = "hitom_CoT"
# # DATASET_TYPE = "fantom_CoT"
# DATASET_TYPE = "fantom_CoT"
# DATASET_TYPE = "bigtom_CoT"
# DATASET_TYPE = "hitom"
# DATASET_TYPE = "fantom"
MAX_WORKERS = 4
def main() -> None:
    summary = run_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        provider=PROVIDER,
        model=MODEL,
        dataset_type=DATASET_TYPE,
        sample_limit=SAMPLE_LIMIT,
        debug=DEBUG,
        max_workers=MAX_WORKERS,
    )
    print("Run finished")
    print(summary)

#
# if __name__ == "__main__":
#     main()




from hitom_llm.main import run_retry_unresolved
# #
# #
# PROJECT_ROOT = Path(__file__).resolve().parent
# DATA_PATH = PROJECT_ROOT / "Hi-ToM_data.json"
# DATA_PATH = PROJECT_ROOT / "conditions/0_forward_belief_true_belief/stories.csv"
# DATA_PATH = PROJECT_ROOT / "fantom.json"

# PREVIOUS_OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_bigtom_true_belief_fix"
# NEW_OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_bigtom_true_belief_fix_unresolved"
#
# PREVIOUS_OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_fantom_resolved_resolved_resolved_resolved_gemini_3"
# NEW_OUTPUT_DIR = PROJECT_ROOT / "results_pycharm_fantom_resolved_resolved_resolved_resolved_resolved_gemini_3"
#
#
# # PROVIDER = "openai"
# # MODEL = "gpt-5.4-nano"
# # MODEL = "gpt-5.4"
# PROVIDER = "gemini"
# MODEL = "gemini-3-flash-preview"
# # DATASET_TYPE = "bigtom"
# DATASET_TYPE = "fantom"
# SAMPLE_LIMIT = 640
# DEBUG = True
# #
# MAX_WORKERS = 4
# def main() -> None:
#     summary = run_retry_unresolved(
#         data_path=DATA_PATH,
#         previous_output_dir=PREVIOUS_OUTPUT_DIR,
#         new_output_dir=NEW_OUTPUT_DIR,
#         provider=PROVIDER,
#         model=MODEL,
#         dataset_type=DATASET_TYPE,
#         sample_limit=SAMPLE_LIMIT,
#         debug=DEBUG,
#         max_workers=MAX_WORKERS,
#     )
#     print("Retry unresolved finished")
#     print(summary)


if __name__ == "__main__":
    main()