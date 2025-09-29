"""Standalone Bash code execution tool extracted from GenomeAgent.

This module consolidates the bash execution logic (shell runner, executor,
validation rules, and tool wrapper) into a single file so it can be copied to
other projects without needing the rest of the GenomeAgent package.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import stat
import subprocess
import sys
import time
from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

import psutil
import requests
from ..config import get_env

EASY_RUN_TOOL_NAMES: Sequence[str] = ("bismark", "cellranger", "flye", "STAR")

LOG_SKIP_STRINGS: Tuple[str, ...] = (
    "libmamba Cache",
    "conda info --envs",
    "EnvironmentNameNotFound",
    "Using cache",
    "Transaction",
    "Prefix: ",
    "All requested packages already installed",
    "warning  libmamba",
    "Checked",
    "check zst",
    "/linux-64",
    "/noarch",
    "─" * 10,
    "B  conda-forge",
    "The following NEW packages will be INSTALLED",
    "The following packages will be downloaded",
    "Collecting package metadatas",
    "Solving environment",
    "## Package Plan ##",
    "environment location",
    "updated specs",
    "was built under R version",
    "Loading required package",
    "Attaching package:",
    "The following objects are masked",
    "WARNING",
)

INTERACTIVE_EDITORS: Tuple[str, ...] = ("nano", "vi", "vim")

def print_colored(text, color=None, do_print=True):
    """Print coloured text with an optional fallback implementation."""

    color_dict = {
        "BLUE": "\033[34m",
        "LIGHT_BLUE": "\033[94m",
        "ORANGE": "\033[38;5;214m",
        "GREEN": "\033[32m",
        "LIGHT_GREEN": "\033[92m",
        "RED": "\033[31m",
        "LIGHT_RED": "\033[91m",
        "YELLOW": "\033[33m",
        "LIGHT_YELLOW": "\033[93m",
        "MAGENTA": "\033[35m",
        "LIGHT_MAGENTA": "\033[95m",
        "CYAN": "\033[36m",
        "LIGHT_CYAN": "\033[96m",
        "WHITE": "\033[37m",
        "LIGHT_WHITE": "\033[97m",
        "BLACK": "\033[30m",
        "GRAY": "\033[90m",
        "DARK_GRAY": "\033[38;5;236m",
        "BOLD": "\033[1m",
        "UNDERLINE": "\033[4m",
        "RESET": "\033[0m",
    }

    if isinstance(text, dict):
        text = json.dumps(text).replace(r"\n", "\n")

    if isinstance(color, int) and 0 <= color <= 255:
        ansi_color = f"\033[38;5;{color}m"
        print(f"{ansi_color}{text}{color_dict['RESET']}", flush=True)
    elif isinstance(color, str) and do_print:
        print(f"{color_dict[color]}{text}{color_dict['RESET']}", flush=True)
    elif not color and do_print:
        print(text, flush=True)
    elif isinstance(color, str) and not do_print:
        return f"{color_dict[color]}{text}{color_dict['RESET']}"
    else:
        return None
    return None


# ---------------------------------------------------------------------------
# Shell runner implementation (originally smolagents.genomeagent_module.shell)
# ---------------------------------------------------------------------------


def has_printable_chars(string):
    pattern = re.compile(r"[^\s]")
    return bool(pattern.search(string))


def _remove_col_row_tokens(content: str) -> str:
    return re.sub(r"\b(col|row)\w+\b,?\s*", "", content)


def _has_meaningful_text(content: str) -> bool:
    return not bool(re.match(r"^\s*$|^[^a-zA-Z0-9]*$", content))


def _should_skip_log_line(content: str) -> bool:
    if contains_any(content, LOG_SKIP_STRINGS):
        return True
    stripped = content.strip()
    if not stripped:
        return True
    if not _has_meaningful_text(stripped):
        return True
    if stripped.endswith("Cached"):
        return True
    return False


def _contains_any_tool(command_text: str, tools: Iterable[str]) -> bool:
    lowered = command_text.lower()
    return any(re.search(rf"\\b{re.escape(tool.lower())}\\b", lowered) for tool in tools)


def get_process_tree(proc):
    procs = [proc]
    try:
        children = proc.children(recursive=True)
        procs.extend(children)
    except psutil.NoSuchProcess:
        time.sleep(1)
    return procs


def monitor_process_tree(procs):
    total_cpu = 0.0
    total_read_bytes = 0.0
    total_write_bytes = 0.0
    for p in procs:
        try:
            total_cpu += p.cpu_percent(interval=0.1)
            io_counters = p.io_counters()
            total_read_bytes += io_counters.read_bytes
            total_write_bytes += io_counters.write_bytes
        except psutil.NoSuchProcess:
            time.sleep(1)
    return total_cpu, total_read_bytes, total_write_bytes


def read_nonblocking(stream):
    output = []
    while True:
        ready, _, _ = select.select([stream], [], [], 0)
        if not ready:
            break
        line = stream.readline()
        if not line:
            break
        output.append(line)
    return "\n".join(output)


def kill_proc_tree(pid, sig=signal.SIGTERM, include_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.send_signal(sig)
    psutil.wait_procs(children, timeout=5)
    if include_parent:
        try:
            parent.send_signal(sig)
        except Exception:
            time.sleep(1)
        parent.wait(5)


def watch(instance, proc_psutil):
    while True:
        if instance.process.poll() is not None:
            break

        monitor_process(instance, proc_psutil)

        if check_termination_conditions(instance):
            terminate_process(instance)
            time.sleep(10)
            break

        handle_process_output(instance)
        time.sleep(instance.interval)

    if instance.terminate and instance.kill == "psutil" and instance.process.poll() is not None:
        instance.process.terminate()
        time.sleep(10)

    handle_process_output(instance)
    return finalize_process(instance)


def monitor_process(instance, proc_psutil):
    procs = get_process_tree(proc_psutil)
    cpu_usage, read_bytes, write_bytes = monitor_process_tree(procs)
    read_megabytes, write_megabytes = read_bytes / (1024 ** 2), write_bytes / (1024 ** 2)

    instance.cpu_record.append(cpu_usage <= instance.cpu_threshold)
    instance.read_record.append(read_megabytes <= instance.cpu_threshold)
    instance.write_record.append(write_megabytes <= instance.cpu_threshold)

    if instance.show and instance.do_exit:
        print(
            f"[stdout] CPU usage {cpu_usage:.2f}% IO read {read_megabytes:.2f} MBs write {write_megabytes:.2f} MBs",
            flush=True,
        )


def check_termination_conditions(instance):
    if sum(instance.cpu_record) == instance.threshold:
        return True
    if instance.read_threshold and instance.read_threshold > 0 and sum(instance.read_record) == instance.threshold:
        return True
    if instance.write_threshold and instance.write_threshold > 0 and sum(instance.write_record) == instance.threshold:
        return True
    return False


def terminate_process(instance):
    instance.terminate = True
    if instance.do_exit:
        print("[stderr] Terminating bash code process due to its inactivity.")
    if instance.kill == "subprocess":
        instance.process.terminate()
    else:
        kill_proc_tree(instance.process.pid)


def handle_process_output(instance):
    stdout_batch = read_nonblocking(instance.process.stdout)
    stderr_batch = read_nonblocking(instance.process.stderr)
    instance.print_chars(stdout_batch, type="stdout")
    instance.print_chars(stderr_batch, type="stderr")


def finalize_process(instance):
    flag = False if instance.terminate or instance.process.returncode != 0 else True
    if not instance.do_exit:
        return instance.stdout_data, instance.stderr_data, flag
    if not flag:
        instance.process.returncode = 1 if instance.process.returncode == 0 else instance.process.returncode
        print(f"[stderr] Bash codes finish with exit code: {instance.process.returncode}")
        if not instance.llm_test:
            sys.exit(1)
    if instance.process.returncode == 0:
        print("[stdout] Bash codes executed successfully!")
    return instance.stdout_data, instance.stderr_data, flag


class ShellRunner:
    def __init__(self, script, times, interval, cpu, read, write, show, kill, do_exit, llm_test=False):
        self.script = script
        self.threshold, self.interval = times, interval
        self.cpu_record, self.read_record, self.write_record = (
            deque(maxlen=self.threshold),
            deque(maxlen=self.threshold),
            deque(maxlen=self.threshold),
        )
        self.cpu_threshold, self.read_threshold, self.write_threshold = cpu, read, write
        self.stdout_data = []
        self.stderr_data = []
        self.terminate = False
        self.process = None
        self.show = show
        self.kill = kill
        self.do_exit = do_exit
        self.llm_test = llm_test
        self.easyrun_tools = list(EASY_RUN_TOOL_NAMES)

    def print_chars(self, content, type="stdout"):
        target_data = self.stdout_data if type == "stdout" else self.stderr_data
        for line in content.split("\n"):
            stripped_line = line.strip()
            if has_printable_chars(stripped_line):
                target_data.append(stripped_line)
                if self.do_exit:
                    print(f'[{type}] ' + stripped_line, flush=True)

    def easyrun(self, cmd):
        sr = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.stdout_data, self.stderr_data, returncode = sr.stdout, sr.stderr, sr.returncode == 0

        flag = False if self.terminate or returncode != 0 else True
        if not self.do_exit:
            return self.stdout_data, self.stderr_data, flag
        if not flag:
            returncode = 1 if returncode == 0 else returncode
            print(f"[stderr] Bash codes finish with exit code: {returncode}")
            if not self.llm_test:
                sys.exit(1)
        if returncode == 0:
            print("[stdout] Bash codes executed successfully!")
        return self.stdout_data, self.stderr_data, returncode == 0

    def processor(self):
        cmd = ['/bin/bash', self.script]

        with open(self.script, 'r') as f:
            bash_content = f.read().lower()
        if _contains_any_tool(bash_content, self.easyrun_tools):
            return self.easyrun(cmd)
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        proc_psutil = psutil.Process(self.process.pid)
        return watch(self, proc_psutil)


# ---------------------------------------------------------------------------
# Agent executor implementation (originally genomeagent_module.executor)
# ---------------------------------------------------------------------------

Easy_Run_tools = list(EASY_RUN_TOOL_NAMES) + ['fastqc']

OPENMS_tools = ["omssacl", "omssa", "TOPPView", "TOPPAS", "INIFileEditor", "SwathWizard", "FLASHDeconvWizard",
    "File Converter", "FileConverter", "GNPSExport", "IDFileConverter", "MSstatsConverter", "MzTabExporter",
    "TargetedFileConverter", "TextExporter", "TriqlerConverter", "File Filtering", "DatabaseFilter", "DecoyDatabase",
    "DTAExtractor", "FileFilter", "FileInfo", "FileMerger", "IDFilter", "IDMerger", "IDRipper", "IDSplitter",
    "IonMobilityBinning", "MapStatistics", "MzMLSplitter", "PeakPickerHiRes", "PeakPickerIterative", "BaselineFilter",
    "NoiseFilterGaussian", "NoiseFilterSGolay", "MapNormalizer", "SpectraFilterNLargest", "SpectraFilterNormalizer",
    "SpectraFilterThresholdMower", "SpectraFilterWindowMower", "MaRaClusterAdapter", "Resampler", "SpectraMerger",
    "InternalCalibration", "ExternalCalibration", "HighResPrecursorMassCorrector", "IDRTCalibration", "Quantitation",
    "ConsensusMapNormalizer", "Decharger", "EICExtractor", "FeatureFinderCentroided", "FeatureFinderIdentification",
    "FeatureFinderMetabo", "FeatureFinderMetaboIdent", "FeatureFinderMultiplex", "IsobaricAnalyzer",
    "MassTraceExtractor", "MetaboliteAdductDecharger", "MetaProSIP", "MultiplexResolver", "ProteinQuantifier",
    "ProteomicsLFQ", "SeedListGenerator", "CometAdapter", "LuciphorAdapter", "MascotAdapter", "MascotAdapterOnline",
    "MSFraggerAdapter", "MSGFPlusAdapter", "NovorAdapter", "SageAdapter", "SimpleSearchEngine", "SpecLibSearcher",
    "SpectraSTSearchAdapter", "XTandemAdapter", "Identification Processing", "ConsensusID", "Digestor", "DigestorMotif",
    "Epifany", "FalseDiscoveryRate", "IDConflictResolver", "IDDecoyProbability", "IDExtractor", "IDMapper",
    "IDMassAccuracy", "IDPosteriorErrorProbability", "IDScoreSwitcher", "PeptideIndexer", "PercolatorAdapter",
    "PhosphoScoring", "ProteinInference", "PSMFeatureExtractor", "SequenceCoverageCalculator", "SpecLibCreator",
    "StaticModification", "MapAlignerIdentification", "MapAlignerPoseClustering", "MapAlignerTreeGuided",
    "MapRTTransformer", "FeatureLinkerLabeled", "FeatureLinkerUnlabeled", "FeatureLinkerUnlabeledQT",
    "FeatureLinkerUnlabeledKD", "AssayGeneratorMetabo", "AssayGeneratorMetaboSirius", "ClusterMassTracesByPrecursor",
    "MRMMapper", "MRMPairFinder", "MRMTransitionGroupPicker", "OpenSwathAnalyzer", "OpenSwathAssayGenerator",
    "OpenSwathChromatogramExtractor", "OpenSwathConfidenceScoring", "OpenSwathDecoyGenerator", "OpenSwathDIAPreScoring",
    "OpenSwathFeatureXMLToTSV", "OpenSwathFileSplitter", "OpenSwathMzMLFileCacher", "OpenSwathRewriteToFeatureXML",
    "OpenSwathRTNormalizer", "OpenSwathWorkflow", "OpenPepXL", "OpenPepXLLF", "RNPxlXICFilter", "XFDR", "FLASHDeconv",
    "QualityControl", "DatabaseSuitability", "QCCalculator", "QCEmbedder", "QCExporter", "QCExtractor", "QCImporter",
    "QCMerger", "QCShrinker", "AccurateMassSearch", "MetaboliteSpectralMatcher", "SiriusExport",
    "NucleicAcidSearchEngine", "RNADigestor", "RNAMassCalculator", "ClusterMassTraces", "DeMeanderize",
    "ExecutePipeline", "GenericWrapper", "ImageCreator", "INIUpdater", "MassCalculator", "MetaProSIP", "OpenMSInfo",
    "TICCalculator", "CVInspector", "FuzzyDiff", "JSONExporter", "OpenMSDatabasesInfo", "SemanticValidator",
    "XMLValidator"]

class AgentCodeExecutor:
    def __init__(
            self,
            running_env='local',
            black_list=[],
            conda_home='/Users/liguowei/ubuntu/miniconda3',
            conda_bioenv='bioenv',
            conda_renv='bioenv',
            max_log_len=4096,
            max_stdout_line=30,
            max_stderr_line=80,
            EIFlow=None,
            debug=False,

    ):
        # log setting
        self.debug = debug
        self.max_log_len = max_log_len
        self.max_stdout_line = max_stdout_line
        self.max_stderr_line = max_stderr_line

        self.running_env = running_env
        self.black_list = black_list

        # conda setting
        self.conda_home = conda_home
        self.conda_bioenv = conda_bioenv
        self.conda_renv = conda_renv
        self.conda_path = None
        if self.conda_home:
          self.conda_path = os.path.join(self.conda_home, "./bin/conda")

        self.bashcode_prefix: List[str] = []
        if running_env == 'local':
            self.bashcode_prefix = [
                '#!/bin/bash',
                'set -e',
                # 'shopt -s expand_aliases',
                # f"export MAMBA_ROOT_PREFIX={self.conda_home}",
                # f"export PATH={self.conda_home}bin:{self.conda_home}envs/{self.conda_bioenv}/bin:{self.conda_home}envs/{self.conda_renv}/bin:" + "${PATH}",
                # 'eval "$(conda shell.bash hook)"',
                'echo "channels:',
                "  - conda-forge",
                "  - bioconda",
                "  - defaults",
                "channel_priority: strict",
                'ssl_verify: false"  >  ~/.condarc',
                ]
        if get_env("http_proxy_port"):
            http_proxy_port = get_env("http_proxy_port")
            self.bashcode_prefix.extend([
                f"conda config --set proxy_servers.http {http_proxy_port}",
                f"conda config --set proxy_servers.https {http_proxy_port}",
                f"export http_proxy={http_proxy_port}",
                f"export https_proxy={http_proxy_port}",
            ])

    def conda_search(self, package_name=None, conda_name=None):
        if not package_name or not conda_name:
            return []
        package_info = {}
        try:
            # 使用 subprocess 运行 conda search 命令
            result = subprocess.run([f'mamba search', package_name],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout
            if result.returncode != 0:
                print_colored(f"Error searching for package: {result.stderr}", 'RED', self.debug)
                return None
            flag = False
            # 解析输出，找到包的名字和渠道信息
            for line in output.splitlines():
                if 'Name' in line and 'Version' in line and 'Channel' in line:
                    flag = True
                    continue
                if flag:
                    package, channel = line.split()[0], line.split()[-1]
                    package_info[package] = channel
        except Exception as e:
            print_colored(f"An error occurred: {e}", 'RED', self.debug)
            return None
        install_command = []
        if package_info:
            for pkg, channel in package_info.items():
                install_command.append(f"mamba install -c {channel} -n {conda_name} {pkg}")
        else:
            print_colored(f"Package {package_name} not found.", 'RED', self.debug)
        return install_command

    def conda_list(self, conda_home=None, conda_env=None):
        results = []
        list_file_path = f"{conda_home}/{conda_env}.list"
        if os.path.exists(list_file_path):
            with open(list_file_path, "r") as f:
                result = f.read()
            results = result.splitlines()
        else:
            result = subprocess.run([f'conda', 'list', '-n', conda_env],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            results = result.stdout.splitlines()
            # 将结果写入文件
            with open(list_file_path, "w") as f:
                f.write(result.stdout)
        output = []
        for i in results:
            if i.strip():
                output.append(i.split()[0])
        return output

    def conda_install_cmd_replace(self, bash_lines, installed, pseudo='fastqc', remove=False):
        CMDS = bash_lines
        # 初始化一个结果列表
        packages = []
        final_cmds = []
        for origin_cmd in CMDS:
            cmd = origin_cmd
            if ('conda' in cmd or 'mamba' in cmd or 'pip' in cmd) and ('install' in cmd or 'create' in cmd):
                tokens = cmd.split()
                # 标志变量，用于忽略源选项（例如 -c 或 --channel）
                skip_next = False
                for token in tokens:
                    # 如果前一个标志要求跳过，直接跳过（例如 -c conda-forge 的 conda-forge）
                    if skip_next:
                        skip_next = False
                        continue
                    # 跳过选项和命令关键字
                    if token in {"conda", "mamba", "install", "-y", "--yes"}:
                        continue
                    # 如果是选项（如 -c 或 --channel），则跳过该选项并标记下一个内容
                    if token in {"-c", "--channel"}:
                        skip_next = True
                        continue
                    # 如果通过了所有过滤条件，认为是包名
                    packages.append(token)
                    # 如果已安装，则替换命令
                    if token.lower() in installed:
                        cmd = cmd.replace(token, pseudo)
                if not remove:
                    final_cmds.append(cmd)
            else:
                final_cmds.append(cmd)
        return "\n".join(final_cmds)

    def modify_code(self, bash_content):
        # replace conda related cmds
        # replace conda related cmds
        bash_content_raw = bash_content
        bash_content = re.sub(r'CONDA_NO_PLUGINS=true', '', bash_content)
        bash_content = re.sub(r'(?:^|\b)conda\b', 'mamba', bash_content)
        bash_content = re.sub(r'mamba-forge', 'conda-forge', bash_content)
        bash_content = re.sub(r'conda install', 'mamba install', bash_content)
        bash_content = re.sub(r'minimamba', 'miniconda', bash_content)
        bash_content = re.sub(r'source activate', 'conda activate', bash_content)
        bash_content = re.sub(r'mamba activate', 'conda activate', bash_content)

        # split bash code to lines
        # 先查找第一个出现的 Rscript 或 python 命令，并提取出来
        sub_command = ''
        script_match_1 = re.search(
            r' (Rscript|python|awk|perl|sed) .*',
            bash_content, re.DOTALL)
        script_match_2 = re.search(
            r'^(Rscript|python|awk|perl|sed) .*',
            bash_content, re.DOTALL)
        script_match_3 = re.search(
            r'\n(Rscript|python|awk|perl|sed) .*',
            bash_content, re.DOTALL)
        if script_match_1:
            sub_command = script_match_1.group(0)
            bash_content = bash_content[:script_match_1.start()]
        elif script_match_2:
            sub_command = script_match_2.group(0)
            bash_content = bash_content[:script_match_2.start()]
        elif script_match_3:
            sub_command = script_match_3.group(0)
            bash_content = bash_content[:script_match_3.start()]
        else:
            bash_content, sub_command = bash_content, ''
        bash_lines = [command for command in re.split(r'(?:&& |&& \\|\s*;\s*|\n)', bash_content)]
        bash_lines.append(sub_command)

        # process installed tools and Rscript related tools
        if any(keyword in bash_content_raw for keyword in ['Rscript', 'requireNamespace', 'install.packages', '::install', 'BiocManager']):
            bash_lines = [f"conda activate {self.conda_renv}"] + bash_lines
            # skip all conda install tools when using R
            installed_list = self.conda_list(conda_home=self.conda_home, conda_env=self.conda_renv)
            bash_content = self.conda_install_cmd_replace(bash_lines, installed_list, pseudo='fastqc', remove=True)
            for i in self.conda_search(package_name=None, conda_name=self.conda_renv):
                self.bashcode_prefix.append(i)
            self.bashcode_prefix.append(f"conda activate {self.conda_renv}")
        else:
            bash_lines = [f"conda activate {self.conda_bioenv}"] + bash_lines
            # skip installed packages
            installed_list = self.conda_list(conda_home=self.conda_home, conda_env=self.conda_bioenv)
            bash_content = self.conda_install_cmd_replace(bash_lines, installed_list, pseudo='fastqc', remove=True)
            for i in self.conda_search(package_name=None, conda_name=self.conda_bioenv):
                self.bashcode_prefix.append(i)
        for openms in OPENMS_tools:
            if openms.lower() in bash_content.lower():
                if "conda activate" in bash_content:
                    bash_content = re.sub(r' bioenv.*?(\s)', r' protein\1', bash_content)
                else:
                    bash_content = "conda activate protein\n" + bash_content
            break
        return bash_content

    def write_modify_code(self, job_shell):
        job_shell_path = job_shell + ".exe"
        with open(job_shell, "r") as f:
            bash_content = f.read()
        modify_content = self.modify_code(bash_content)

        modify_content = re.sub(r'bioenv(?!_)', f'{self.conda_bioenv}', modify_content)
        modify_content = re.sub(r'Renv(?!_)', f'{self.conda_renv}', modify_content)
        print_colored("\n```bash", 'GREEN', self.debug)
        print_colored(modify_content, 'GREEN', self.debug)
        print_colored("```\n", 'GREEN', self.debug)

        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        fd = os.open(job_shell_path, flags, modes)

        with os.fdopen(fd, "w") as w:
            for code in self.bashcode_prefix:
                w.write(code + "\n")
            w.write(modify_content)

    def write_execute(self, job_shell=None):
        # 本地job_shell增加前置信息
        # 本地job_shell增加前置信息
        if not job_shell.startswith("/"):
            self.job_shell = job_shell.lstrip("./")
        else:
            self.job_shell = job_shell
        self.write_modify_code(self.job_shell)

    # 任务状态检查器，检查作业成功与否，以及获取日志转储链接，读取日志内容
    def job_status_check(self, eiflow_job_name=None):
        if self.running_env == 'local':
            # 本地任务，直接等待程序执行结果，并返回处理log
            job_shell_path = self.job_shell + ".exe"
            stdout, stderr, execute_statu, bash_content = None, None, None, None
            with open(self.job_shell, 'r') as f:
                bash_content = f.read().lower()
            if contains_any(bash_content, Easy_Run_tools):
                stdout, stderr, execute_statu = easyrun(job_shell_path)
                # print_colored(f"stdout, stderr, execute_statu: {execute_statu}", "ORANGE")
            else:
                # use ShellRunner class in shell.py module
                sr = ShellRunner(script=job_shell_path, times=180, interval=3, cpu=0, read=None, write=None, show=False, kill='psutil', do_exit=False)
                stdout, stderr, execute_statu = sr.processor()
            log_info = []
            log_info.extend(filter_content(stdout, "stdout"))
            log_info.extend(filter_content(stderr, "stderr"))
            _, _, _, executor_info = log_processor(log_info, self.max_stdout_line, self.max_stderr_line, self.max_log_len, path_prefix_to_remove=None, timestamp_length=None)
        else:
            raise ValueError(f"Unknown running environment: {self.running_env}")

        # print_colored(f"if execute_statu: {execute_statu}", "ORANGE")
        if execute_statu:
            return executor_info, True
        else:
            return executor_info, False

def easyrun(job_shell_path):
    # 本地任务，直接等待程序执行结果，并返回处理log
    # do not use ShellRunner class
    # print_colored("using subprocess easy run mode", "RED")
    sr = subprocess.run(['/bin/bash', job_shell_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print_colored(f"easyrun sr.returncode={sr.returncode}", "ORANGE")
    stdout, stderr, execute_statu = sr.stdout.splitlines(), sr.stderr.splitlines(), sr.returncode == 0
    return stdout, stderr, execute_statu

def contains_any(string, substrings):
    return any(substring in string for substring in substrings)

def delete_file(self, f):
    if os.path.exists(f):
        os.remove(f)

def debug_log(file, content):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(str(content))

def log_processor(logs, max_stdout_line, max_stderr_line, max_log_len, path_prefix_to_remove=None, timestamp_length=None):
    processed_logs, stdout, stderr = [], [], []
    # timestamp_length = 20  # 时间戳的长度， 例如'2024-04-16T08:04:23Z'
    for line in logs:
        # 删除时间戳、[stdout]与[stderr]
        # timestamp_length = min(timestamp_length, min(line.find('[stderr]'), line.find('[stdout]')))
        if timestamp_length and len(line) > timestamp_length:
            line = line[timestamp_length:].strip().lstrip()
        # 检查是否存在指定的路径前缀，如果存在，删除
        if path_prefix_to_remove and line.startswith(path_prefix_to_remove):
            line = line.replace(path_prefix_to_remove, "")
        # 删除空白行 或者 全为空白符、Unicode字符 或者 不含含义的行
        if (not line) or re.compile(r'\s*').fullmatch(line) or (not re.search(r'[a-zA-Z0-9]', line)) or re.compile(r'(\\u[0-9a-fA-F]{4})+').fullmatch(line):
            continue
        processed_logs.append(line)
        # 独立保留stderr和stdout开头的行（与genome-env的shell.py执行脚本功能相匹配），单独保存stderr和stdout队列。
        if line.startswith("[stdout]"):
            # if_skip, content = self.filter_stdout(line)
            # if not if_skip:
            stdout.append(line)
        if line.startswith("[stderr]"):
            # if_skip, content = self.filter_stderr(line)
            # if not if_skip:
            stderr.append(line)

    if len(stdout) > max_stdout_line:
        mid = int(len(stdout) / 2) if int(len(stdout) / 2) <= int(max_stdout_line/2) else int(max_stdout_line/2)
        stdout = stdout[:mid] + ["......", "......", "......"] + stdout[-mid:]
    if len(stderr) > max_stderr_line:
        mid = int(len(stderr) / 2) if int(len(stderr) / 2) <= int(max_stderr_line/2) else int(max_stderr_line/2)
        stderr = stderr[:mid] + ["......", "......", "......"] + stderr[-mid:]

    stdout = '\n'.join(stdout)
    stderr = '\n'.join(stderr)
    if len(stderr) + len(stdout) < max_log_len:
        executor_info = stdout + '\n' + stderr
    elif len(stderr) > max_log_len:
        executor_info = stdout + '\n' + stderr[:int(max_log_len/2)] + "......" + stderr[-int(max_log_len/2):]
    elif len(stderr) + len(stdout) > max_log_len:
        while len(stderr) + len(stdout) > max_log_len:
            stdout = stdout[:int(len(stdout) / 4)] + "......" + stdout[-int(len(stdout) / 4):]
            stderr = stderr[:int(len(stderr) / 3)] + "......" + stderr[-int(len(stderr) / 3):]
        executor_info = stdout + '\n' + stderr
    else:
        executor_info = stdout + '\n' + stderr
    return processed_logs, stdout, stderr, executor_info

def filter_content(content, type):
    res = []
    for i in content:
        if _should_skip_log_line(i):
            continue
        if sum([x.startswith("col") or x.startswith("row") for x in i.split(", ")]) >= 3:
            continue
        res.append(f'[{type}] ' + i.strip())
    return res


def filter_stdout(content):
    cleaned = _remove_col_row_tokens(content)
    if _should_skip_log_line(cleaned):
        return True, content
    return False, cleaned


def filter_stderr(content):
    cleaned = _remove_col_row_tokens(content)
    if _should_skip_log_line(cleaned):
        return True, content
    return False, cleaned


def code_rule_sudo(bash_code: str, debug: bool = False) -> [bool, str]:
    if "sudo " in bash_code:
        print_colored(f'### Code Execute Statu: Failed, log: \n```using sudo is not allowed!\n```\n', 'ORANGE', debug)
        return False,f'Code Execute Statu: Failed! The failed log: using sudo is not allowed!'
    return True, None

def code_rule_file_not_found(bash_code: str, debug: bool = False) -> [bool, str]:
    if "if [ -f " in bash_code or " ! -f " in bash_code or "[ -f " in bash_code or "[[ -f " in bash_code:
        print_colored(
            f'### Code Execute Statu: Failed, log: \n```Check file exist is not allowed! "No such file or directory" '
            f'means error in previous codes!\n```\n',
            'ORANGE', debug)
        return False, (f'Code Execute Statu: Failed! The failed log: Check file exist is not allowed! "No such file or '
                f'directory" means error in previous steps!')
    return True, None

def code_rule_multiple_r_script(bash_code: str, debug: bool = False) -> [bool, str]:
    if bash_code.lower().count("rscript -e") >= 2:
        print_colored(f'Code Execute Statu: Failed! The failed log: Multiple Rscript commands must be merged in one command!', 'ORANGE', debug)
        return False, f'Code Execute Statu: Failed! The failed log: Multiple Rscript commands must be merged in one command!'
    return True, None

def code_rule_multiple_python_script(bash_code: str, debug: bool = False) -> [bool, str]:
    if bash_code.lower().count("python -c") >= 2 or bash_code.lower().count(
            "python2 -c") >= 2 or bash_code.lower().count("python3 -c") >= 2:
        print_colored(f'Code Execute Statu: Failed! The failed log: Multiple python commands must be merged in one command!', 'ORANGE', debug)
        return False, f'Code Execute Statu: Failed! The failed log: Multiple python commands must be merged in one command!'
    return True, None

def code_rule_interactive_tool(bash_code: str, debug: bool = False) -> [bool, str]:
    for editor in INTERACTIVE_EDITORS:
        pattern = rf'(^|[^a-zA-Z0-9]){editor} '
        if re.search(pattern, bash_code):
            if re.search(rf'-{editor} ', bash_code):
                continue
            print_colored(
                'Code Execute Statu: Failed! Using "nano"/"vi"/"vim" is not allowed in a non-interactive environment!',
                'ORANGE',
                debug,
            )
            return False, 'Code Execute Statu: Failed! Using "nano"/"vi"/"vim" is not allowed in a non-interactive environment!'
    return True, None


Code_Rule_Functions = {
    'code_rule_sudo': code_rule_sudo,
    'code_rule_file_not_found': code_rule_file_not_found,
    'code_rule_multiple_r_script': code_rule_multiple_r_script,
    'code_rule_multiple_python_script': code_rule_multiple_python_script,
    'code_rule_interactive_tool': code_rule_interactive_tool,
}

def code_whitelist_deseq_no_count(execute_log):
    if "all samples have 0 counts for all genes" in execute_log or "every gene contains at least one zero" in execute_log:
        return True
    return False

def code_whitelist_macs_callpeak(execute_log):
    if ("Done!" in execute_log and "callpeak" in execute_log):
        return True
    return False

def code_whitelist_flye(execute_log):
    if ("INFO: Aligned read sequence:" in execute_log or
            "INFO: Final assembly:" in execute_log or
            "INFO: Assembly statistics" in execute_log or
            "0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%" in execute_log):
        return True
    return False

def code_whitelist_trf(execute_log):
    if "Done." in execute_log and "Resolving output" in execute_log and "Freeing Memory" in execute_log:
        return True
    return False


Code_WhiteList_Log_Functions = {
    "code_whitelist_deseq_no_count": code_whitelist_deseq_no_count,
    "code_whitelist_macs_callpeak": code_whitelist_macs_callpeak,
    "code_whitelist_flye": code_whitelist_flye,
    "code_whitelist_trf": code_whitelist_trf,
}

def pass_WhiteList(execute_log):
    for func_name in Code_WhiteList_Log_Functions.keys():
        statu = Code_WhiteList_Log_Functions.get(func_name)(execute_log)
        if statu:
            return True
    return False


def block_files_in_log(bash_code, execute_log):
    execute_enable_log = []
    if (bool(re.search(r"(?<![a-zA-Z0-9])ls\b", bash_code)) or
        bool(re.search(r"(?<![a-zA-Z0-9])which\b", bash_code)) or
        bool(re.search(r"(?<![a-zA-Z0-9])find\b", bash_code)) or
        bool(re.search(r"(?<![a-zA-Z0-9])head\b", bash_code)) or
        bool(re.search(r"(?<![a-zA-Z0-9])cat\b", bash_code)) or
        bool(re.search(r"(?<![a-zA-Z0-9])awk\b", bash_code)) or
        bool(re.search(r"(?<![a-zA-Z0-9])echo\b", bash_code))):
        for line in execute_log.splitlines():
            i = line.split("/")[-1]
            if i.strip().endswith(".sh") or i.strip().endswith(".rag.log") or i.strip().endswith(".exe") or i.strip().endswith(".exe.log"):
                continue
            execute_enable_log.append(i)
        return True, "\n".join(execute_enable_log)
    return False, None

class BashCodeRunToolWrapper:
    def __init__(
            self,
            debug = False,
            black_list: List[str] = [],
            round_times = 0,
            max_times_per_round: int = 30,
            global_round = 0,
            working_dir = "",
            last_status = False,
            running_env = "local",
            conda_home = '/Users/liguowei/ubuntu/miniconda3',
            conda_bioenv = 'bioenv',
            conda_renv = 'bioenv',
            do_execute = True,
         ):
        self.debug = debug
        self.black_list = black_list
        self.round_times = round_times
        self.max_times_per_round = max_times_per_round
        self.global_round = global_round
        self.working_dir = working_dir
        self.last_status = last_status
        self.running_env = running_env
        self.conda_home = conda_home
        self.conda_bioenv = conda_bioenv
        self.conda_renv = conda_renv
        self.do_execute = do_execute

    def run(self, bash_code: str) -> str:
        """
        This tool receive bash code in string format, then automatically invoke a Linux environment to execute the
        given code, and return a description of the execution status.
        """
        # record times for this round, failed when reaching limitation
        global_round = self.global_round + 1

        # write original code script
        job_shell = os.path.join(self.working_dir, f"step{global_round}.sh" )
        cmd_into_shell_dir = f"cd {self.working_dir}\n"

        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        fd = os.open(job_shell, flags, modes)
        with os.fdopen(fd, "w") as w:
            w.write(cmd_into_shell_dir + bash_code + "\n")

        # code rule check
        for func_name in Code_Rule_Functions.keys():
            statu, log = Code_Rule_Functions.get(func_name)(bash_code)
            if not statu:
                self.last_status = False
                return log

        # print original code
        bash_code = re.sub(re.compile(r"```.*?\n", re.DOTALL), '', bash_code)
        bash_code = re.sub(re.compile(r"```", re.DOTALL), '', bash_code)
        print_colored(f"### code in {job_shell}\n```bash\n{bash_code}\n```", 'GREEN', self.debug)

        # config, modify, and write local code file
        runner = AgentCodeExecutor(running_env=self.running_env,
                                   conda_home=self.conda_home,
                                   conda_bioenv=self.conda_bioenv,
                                   conda_renv=self.conda_renv,
                                   black_list=self.black_list)
        runner.write_execute(job_shell=job_shell)
        # see modify codes
        with open(f"{job_shell}.exe", 'r') as f:
            c = f.readlines()
        if "### prefix end" in c or "### prefix end\n" in c:
            i = c.index("### prefix end\n") if "### prefix end\n" in c else c.index("### prefix end")
            c = "\n".join([x.strip() for x in c[i + 1:]])
        print_colored(f"### execute modified code {job_shell}.exe\n```bash\n{c}\n```", 'GREEN', self.debug)

        # execute code file
        execute_statu = False
        if self.do_execute:
            execute_log, execute_statu = runner.job_status_check()
        else:
            execute_log, execute_statu = f'Code Execute Statu: Success', True

        if 'Rscript' in bash_code:
            execute_log = execute_log.replace("[stderr] ", "").replace("[stdout] ", "")

        # 失败
        if not execute_statu and not pass_WhiteList(execute_log):
            """
            execute_statu	pass_WhiteList  进入条件？
            False	        False           ✅
            False	        True	        ❌
            True	        False	        ❌
            True	        True	        ❌
            """
            # 失败了会计数
            self.last_status = False
            self.round_times += 1
            execute_log = re.sub(re.compile(r"```.*?\n", re.DOTALL), '', execute_log)
            execute_log = re.sub(re.compile(r"```", re.DOTALL), '', execute_log)
            print_colored(f'### Code Execute Statu: Failed {execute_statu}, log: \n```\n{execute_log}\n```\n', 'ORANGE', self.debug)
            return f'Code Execute Statu: Failed! The failed log: {execute_log}'

        # 成功 成功了不计数，因为工具可能会被并行跑
        self.last_status = True
        self.round_times += 0
        log_statu, blocked_log = block_files_in_log(bash_code, execute_log)
        if log_statu:
            print_colored(f'### Code Execute Statu: Success {execute_statu}! Log: \n```\n{blocked_log}\n```\n', 'ORANGE', self.debug)
            return f'Code Execute Statu: Success! Log: {blocked_log}'
        else:
            print_colored(f'### Code Execute Statu: Success {execute_statu}!', 'ORANGE', self.debug)
            return f'Code Execute Statu: Success!'


from smolagents import Tool

class BashCodeRunTool(Tool):
    name = "bash_code_runner"
    description = """Executes Bash commands in an automated Linux environment and returns the result status."""
    inputs = {
        "code": {
            "type": "string",
            "description": "The Bash command or script to execute, e.g., 'ls -l' or 'echo $VAR'.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        debug=False,
        black_list=None,
        round_times=0,
        max_times_per_round=30,
        global_round=0,
        working_dir="/Users/liguowei/ubuntu/virtuallab/genome",
        last_status=False,
        running_env="local",
        conda_home='/Users/liguowei/ubuntu/miniconda3',
        conda_bioenv='bioenv',
        conda_renv='bioenv',
        do_execute=True,
        **kwargs
    ):
        super().__init__()
        if black_list is None:
            black_list = []
        self.runner = BashCodeRunToolWrapper(
            debug=debug,
            black_list=black_list,
            round_times=round_times,
            max_times_per_round=max_times_per_round,
            global_round=global_round,
            working_dir=working_dir,
            last_status=last_status,
            running_env=running_env,
            conda_home=conda_home,
            conda_bioenv=conda_bioenv,
            conda_renv=conda_renv,
            do_execute=do_execute,
            **kwargs
        )

    def forward(self, code: str) -> str:
        # 参数透传已在实例化时完成，这里只传递 code
        result = self.runner.run(code)
        return result

bash_code_run_tool = BashCodeRunTool()

if __name__ == "__main__":
    # 默认实例化
    bash_code_run_tool = BashCodeRunTool()
    # 传递部分参数的实例化样例
    bash_code_run_tool_debug = BashCodeRunTool(debug=True)
    bash_code_run_tool_custom_dir = BashCodeRunTool(working_dir="/tmp/test_bash_tool", max_times_per_round=10)
