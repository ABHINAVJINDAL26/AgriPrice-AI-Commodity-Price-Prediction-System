"""AgriPrice AI end-to-end pipeline runner."""

import os
import subprocess
import sys
import time
from datetime import datetime
import schedule

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(script_relative_path: str, title: str) -> None:
	script_path = os.path.join(BASE_DIR, script_relative_path)
	print(f"\n[START] {title}")
	result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR, check=False)
	if result.returncode != 0:
		raise RuntimeError(f"Step failed: {title}")
	print(f"[DONE]  {title}")


def run_pipeline() -> None:
	os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
	os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

	print("=" * 58)
	print("AgriPrice AI - Starting Full Pipeline")
	print("=" * 58)

	run_step("src/data_preprocessing.py", "Data Preprocessing")
	run_step("src/eda.py", "Exploratory Data Analysis")
	run_step("src/model.py", "Model Training")
	run_step("src/evaluation.py", "Evaluation and Forecasting")

	print("\n" + "=" * 58)
	print("Pipeline completed successfully.")
	print("Outputs: outputs/  Models: models/")
	print("=" * 58)


def launch_dashboard() -> None:
	app_path = os.path.join(BASE_DIR, "app.py")
	print("Launching dashboard at http://localhost:8501")
	subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], cwd=BASE_DIR, check=False)


def daily_pipeline() -> None:
	print(f"Auto-running pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	run_pipeline()


if __name__ == "__main__":
	if len(sys.argv) > 1:
		cmd = sys.argv[1].lower()
		if cmd == "dashboard":
			launch_dashboard()
		elif cmd == "schedule":
			print("Scheduler started — pipeline will run daily at 06:00 AM")
			schedule.every().day.at("06:00").do(daily_pipeline)
			while True:
				schedule.run_pending()
				time.sleep(1)
		else:
			print(f"Unknown command: {cmd}")
			sys.exit(1)
	else:
		run_pipeline()
