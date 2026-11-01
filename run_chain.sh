#!/bin/bash
cd /home/claude/protguard
export PG_OUT=/home/claude/protguard/artifacts_real
python3 run_all.py 4 2 500 300 > run_all_real.log 2>&1
python3 extra_checks.py > extras_real.log 2>&1
python3 sweep_steer.py > sweep_real.log 2>&1
python3 report.py > report_real.log 2>&1
echo CHAIN_DONE >> run_all_real.log
