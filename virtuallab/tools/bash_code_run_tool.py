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
from typing import Dict, List, Tuple

import psutil
import requests

try:  # pragma: no cover - optional dependency on original utils module
    from .genomeagent_module.utils import print_colored as _external_print_colored
except Exception:  # pragma: no cover - fallback when copying standalone file
    _external_print_colored = None


def print_colored(text, color=None, do_print=True):
    """Print coloured text with an optional fallback implementation."""

    if _external_print_colored is not None:
        return _external_print_colored(text, color=color, do_print=do_print)

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
        self.easyrun_tool = ['bismark', 'cellranger', 'flye', 'STAR']

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
        pattern_list = []
        for tool in self.easyrun_tool:
            p1 = r'(^|\n)' + re.escape(tool) + r'(\s|$)'
            p2 = r'(?<=\S)[ \t]+' + re.escape(tool) + r'(\s|$)'
            p3 = r'\b' + re.escape(tool) + r'\b'
            pattern_list.extend([p1, p2, p3])
        for pattern in pattern_list:
            if re.search(pattern, bash_content):
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

Easy_Run_tools = ['bismark', 'cellranger', 'flye', 'STAR', 'fastqc']

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
            conda_home=None,
            conda_bioenv='bioenv',
            conda_renv='Renv',
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

        # eiflow setting
        self.eiflow = EIFlow
        if self.eiflow is not None:
            if running_env != 'eiflow':
                raise ValueError('EIFlow must be running on eiflow')
            self.eiflow.health_tool = self.eiflow.job_params['health_tool']
            self.eiflow.workflow_id = self.eiflow.workflow_params["workflow_id"]
            self.eiflow.project_name = self.eiflow.job_params['project_name']
            self.eiflow.obs_working_path = self.eiflow.job_params['obs_working_path']
            self.eiflow.obs_loading_path = self.eiflow.job_params['obs_loading_path']
            self.eiflow.job_output_path = self.eiflow.job_params['job_output_path']
            self.eiflow.workflow_task_name = self.eiflow.workflow_params['tasks'][0]['task_name']
            self.bashcode_prefix = [
                '#!/usr/bin/bash',
                'set -e',
                'shopt -s expand_aliases',
                f"cd {self.eiflow.obs_working_path}",
                f'eval "$({self.conda_home}conda shell.bash hook)"',
                'echo "channels:',
                "  - conda-forge",
                "  - bioconda",
                "  - defaults",
                "channel_priority: strict",
                'ssl_verify: false"  >  ~/.condarc',
                "conda activate bioenv"
            ]

        if running_env == 'local':
            # if not check_and_create_conda(conda_path=self.conda_path, conda_home=self.conda_home):
            #     raise ValueError(f'conda is not installed, neither failed to install it automatically')
            # if not check_and_create_conda_env(conda_path=self.conda_path, env_name=self.conda_bioenv):
            #     raise ValueError(f'bioenv is not installed, neither failed to install it automatically')
            # if not check_and_create_conda_env(conda_path=self.conda_path, env_name=self.conda_renv):
            #     raise ValueError(f'Renv is not installed, neither failed to install it automatically')
            self.bashcode_prefix = [
                '#!/usr/bin/bash',
                'set -e',
                'shopt -s expand_aliases',
                f"export MAMBA_ROOT_PREFIX={self.conda_home}",
                f"export PATH={self.conda_home}bin:{self.conda_home}envs/{self.conda_bioenv}/bin:{self.conda_home}envs/{self.conda_renv}/bin:" + "${PATH}",
                'eval "$(conda shell.bash hook)"',
                'echo "channels:',
                "  - conda-forge",
                "  - bioconda",
                "  - defaults",
                "channel_priority: strict",
                'ssl_verify: false"  >  ~/.condarc',
                # "conda config --set proxy_servers.http http://127.0.0.1:7890",
                # "conda config --set proxy_servers.https http://127.0.0.1:7890",
                # "export http_proxy=http://127.0.0.1:7890",
                # "export https_proxy=http://127.0.0.1:7890",
                ]

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
        bash_content = re.sub(r'java -jar picard.jar', 'picard', bash_content)
        bash_content = re.sub(r'flye-modules assemble', 'flye-modules assemble --threads 32', bash_content)
        bash_content = re.sub(r'flye-modules repeat', 'flye-modules repeat --threads 32', bash_content)
        bash_content = re.sub(r'flye-modules contigger', 'flye-modules contigger --threads 32', bash_content)
        bash_content = re.sub(r'flye-modules polisher', 'flye-modules polisher --threads 32', bash_content)

        if "hisat2-build" in bash_content and " -p " not in bash_content:
            bash_content = re.sub(r'hisat2-build', 'hisat2-build -p 16 ', bash_content)
        if "hisat2 " in bash_content and " -p " not in bash_content:
            bash_content = re.sub(r'hisat2 ', 'hisat2 -p 4 ', bash_content)
        if "bwa mem " in bash_content and " -t " not in bash_content:
            bash_content = re.sub(r'bwa mem ', 'bwa mem -t 8 ', bash_content)

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

        # skip black list tools in conda
        for i in ['qualimap', 'openjdk', 'bioconductor-deseq2', 'r-deseq2', 'deseq2', 'r-tidyverse', 'r-biocmanager', 'r-essentials', 'r', 'r-base']:
            for l in range(len(bash_lines)):
                if 'install' in bash_lines[l]:
                    bash_lines[l] = re.sub(rf'(mamba insta.*) {i} (.*)', rf'\1 fastqc \2', bash_lines[l], flags=re.IGNORECASE)

        # process installed tools and Rscript related tools
        if any(keyword in bash_content_raw for keyword in ['Rscript', 'requireNamespace', 'install.packages', '::install', 'BiocManager']):
            bash_lines = [f"conda activate {self.conda_renv}"] + bash_lines
            for i in range(len(bash_lines)):
                content = bash_lines[i]
                if "R --no-save" in content and "echo '" in content:
                    content = content.replace("echo '", "echo 'options(repos = c(CRAN = \"https://mirrors.sustech.edu.cn/CRAN/\"));")
                if "R --no-save" in content and 'echo "' in content:
                    content = content.replace('echo "', 'echo "options(repos = c(CRAN = \'https://mirrors.sustech.edu.cn/CRAN/\'));')
                bash_lines[i] = content
            # skip all conda install tools when using R
            installed_list = self.conda_list(conda_home=self.conda_home, conda_env=self.conda_renv)
            bash_content = self.conda_install_cmd_replace(bash_lines, installed_list, pseudo='fastqc', remove=True)
            # skip tools in R CRAN
            bash_content = re.sub(r'install.packages\(\'(.+?)\'.*?\)', 'install.packages(\'ggplot2\')', bash_content, flags=re.IGNORECASE, count=0)
            bash_content = re.sub(r'install.packages\(\"(.+?)\".*?\)', 'install.packages(\"ggplot2\")', bash_content, flags=re.IGNORECASE, count=0)
            bash_content = re.sub(r'install\(\'(.+?)\'.*?\)', 'install(\'ggplot2\')', bash_content, flags=re.IGNORECASE, count=0)
            bash_content = re.sub(r'install\(\"(.+?)\".*?\)', 'install(\"ggplot2\")', bash_content, flags=re.IGNORECASE, count=0)
            bash_content = re.sub(r"install\(c\(('.*?)\).*?\)", lambda match: "install(c(" + ", ".join(["'ggplot2'"] * len(re.findall(r"'([^']*)'", match.group(1)))) + "))", bash_content, flags=re.IGNORECASE, count=0)
            bash_content = re.sub(r'install\(c\((".*?)\).*?\)', lambda match: 'install(c(' + ', '.join(['"ggplot2"'] * len(re.findall(r'"([^"]*)"', match.group(1)))) + '))', bash_content, flags=re.IGNORECASE, count=0)
            bash_content = re.sub(r'https://cloud.r-project.org/', 'https://mirrors.sustech.edu.cn/CRAN/', bash_content, flags=re.IGNORECASE, count=0)
            bash_content = re.sub(r'BiocManager::install', 'install.packages', bash_content, flags=re.IGNORECASE, count=0)

            # add CRAN source
            if 'Rscript -e "' in bash_content:
                bash_content = bash_content.replace('Rscript -e "', 'Rscript -e "options(repos = c(CRAN = \'https://mirrors.sustech.edu.cn/CRAN/\'));')
                bash_content = re.sub(r"library(enrichR)", "Sys.setenv(CURL_CA_BUNDLE = '/etc/ssl/certs/ca-certificates.crt');Sys.setenv(SSL_CERT_FILE = '/usr/local/share/ca-certificates/www.mountsinai.org.crt');library(enrichR)", bash_content, flags=re.IGNORECASE, count=0)
            if "Rscript -e '" in bash_content:
                bash_content = bash_content.replace("Rscript -e '", "Rscript -e 'options(repos = c(CRAN = \"https://mirrors.sustech.edu.cn/CRAN/\"));")
                bash_content = re.sub(r'library(enrichR)', 'Sys.setenv(CURL_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt");Sys.setenv(SSL_CERT_FILE = "/usr/local/share/ca-certificates/www.mountsinai.org.crt");library(enrichR)', bash_content, flags=re.IGNORECASE, count=0)

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
        if "import scanpy" in bash_content or "import squidpy" in bash_content or "import tangram" in bash_content or "import anndata" in bash_content:
            if "conda activate" in bash_content:
                bash_content = re.sub(r' bioenv.*?(\s)', r' base\1', bash_content)
            else:
                bash_content = "conda activate base\n" + bash_content
        for openms in OPENMS_tools:
            if openms.lower() in bash_content.lower():
                if "conda activate" in bash_content:
                    bash_content = re.sub(r' bioenv.*?(\s)', r' protein\1', bash_content)
                else:
                    bash_content = "conda activate protein\n" + bash_content
            break
        if "medaka" in bash_content:
            if "conda activate" in bash_content:
                bash_content = re.sub(r' bioenv.*?(\s)', r' medaka\1', bash_content)
            else:
                bash_content = "conda activate medaka\n" + bash_content
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
            # for code in [
            #     "export PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/local/lib/pkgconfig",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/quast",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/dapars/bin",
            #     "export PATH=$PATH:/usr/bin/cellranger-4.0.0",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/REDItools2/src/cineca",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/mirdeep2/bin",
            #     "PERL_MB_OPT='--install_base /home/liguowei/perl5';export PERL_MB_OPT",
            #     "PERL_MM_OPT='INSTALL_BASE=/home/liguowei/perl5';export PERL_MM_OPT",
            #     "export PERL5LIB=/root/data/liguowei/GenomeAgent/mirdeep2/lib/perl5",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/mirdeep2/bin",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/CIRI",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/PARalyzer",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/ribotaper/bin",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/pwiz",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/RED-ML/bin",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/cufflinks",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/APAtrap",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/TRF-master/build/src",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/hmmer-3.3.2/build/bin",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/RepeatMasker-master",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/Porechop-master/bin",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/RSEM-1.3.3/build/RSEM/bin",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/SUPPA-2.4",
            #     "export PATH=$PATH:/mnt/data/liguowei/miniconda3/envs/protein/bin",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/MUMmer3",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/mzmine/bin",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/ont-guppy/bin",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/bin",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/miniasm",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/DaPars2/src",
            #     "export PATH=/root/data/liguowei/GenomeAgent/meme-5.5.7/meme/bin:/root/data/liguowei/GenomeAgent/meme-5.5.7/meme/libexec/meme-5.5.7:$PATH",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/Filtlong/bin",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/NextDenovo/bin",
            #     "export PATH=$PATH:/root/data/liguowei/GenomeAgent/NextDenovo",
            #     "export PATH=$PATH:/mnt/data/liguowei/GenomeAgent/hifiasm",
            #     "export PATH=/mnt/data/liguowei/GenomeAgent/salmon-latest_linux_x86_64/bin:$PATH",
            #     "alias circos='conda activate bioenv_circos && /mnt/data/liguowei/miniconda3/envs/bioenv_circos/bin/circos'",
            #     "alias picard='java -jar /mnt/data/liguowei/GenomeAgent/picard.jar'",
            #     "alias pilon='java -Xmx128G -jar /mnt/data/liguowei/GenomeAgent/pilon.jar'",
            #     "alias fusioncatcher='conda activate py27env && /mnt/data/liguowei/GenomeAgent/fusioncatcher/bin/fusioncatcher'",
            #     "alias clipper='conda activate clipper && /mnt/data/liguowei/miniconda3/envs/clipper/bin/clipper'",
            #     "alias 'configManta.py'='/usr/bin/python2 /root/manta/bin/configManta.py'",
            #     "alias 'configManta'='/usr/bin/python2 /root/manta/bin/configManta.py'",
            #     "alias 'bamclipper'='bamclipper.sh'",
            #     "alias 'eventClusterer'='/mnt/data/liguowei/GenomeAgent/SUPPA-2.4/eventClusterer.py'",
            #     "alias 'psiPerGene'='/mnt/data/liguowei/GenomeAgent/SUPPA-2.4/psiPerGene.py'",
            #     "alias 'eventGenerator'='/mnt/data/liguowei/GenomeAgent/SUPPA-2.4/eventGenerator.py'",
            #     "alias 'multipleFieldSelection'='/mnt/data/liguowei/GenomeAgent/SUPPA-2.4/multipleFieldSelection.py'",
            #     "alias 'significanceCalculator'='/mnt/data/liguowei/GenomeAgent/SUPPA-2.4/significanceCalculator.py'",
            #     "alias 'fileMerger'='/mnt/data/liguowei/GenomeAgent/SUPPA-2.4/fileMerger.py'",
            #     "alias 'psiCalculator'='/mnt/data/liguowei/GenomeAgent/SUPPA-2.4/psiCalculator.py'",
            #     "alias 'suppa'='/mnt/data/liguowei/GenomeAgent/SUPPA-2.4/suppa.py'",
            #     "alias 'CIRI_DE'='conda activate /root/data/liguowei/GenomeAgent/CIRIquant_env && CIRI_DE'",
            #     "alias 'CIRI_DE_replicate'='conda activate /root/data/liguowei/GenomeAgent/CIRIquant_env && CIRI_DE_replicate'",
            #     "alias 'CIRIquant'='conda activate /root/data/liguowei/GenomeAgent/CIRIquant_env && CIRIquant'",
            #     "alias 'prep_CIRIquant'='conda activate /root/data/liguowei/GenomeAgent/CIRIquant_env && prep_CIRIquant'",
            #     "alias 'snpEff'='/usr/bin/java -jar /root/data/liguowei/GenomeAgent/snpEff/snpEff.jar'",
            #     "alias 'SnpSift'='/usr/bin/java -jar /root/data/liguowei/GenomeAgent/snpEff/SnpSift.jar'",
            #     "alias 'snpeff'='/usr/bin/java -jar /root/data/liguowei/GenomeAgent/snpEff/snpEff.jar'",
            #     "alias 'snpsift'='/usr/bin/java -jar /root/data/liguowei/GenomeAgent/snpEff/SnpSift.jar'",
            #     "alias quast='/root/data/liguowei/GenomeAgent/quast/quast.py'",
            #     "alias 'quast.py'='/root/data/liguowei/GenomeAgent/quast/quast.py'",
            #     "alias FusionInspector='conda activate fusioninspector && FusionInspector'",
            #     "alias busco='busco --offline'",
            #     "alias HiC-Pro='conda activate /root/data/liguowei/GenomeAgent/HiC-Pro-master/HiC-Pro && /mnt/data/liguowei/GenomeAgent/bin/HiC-Pro_3.1.0/bin/HiC-Pro'",
            #     "alias taco='/mnt/data/liguowei/miniconda3/envs/py27env/bin/taco_run'",
            #     "alias taco_run='/mnt/data/liguowei/miniconda3/envs/py27env/bin/taco_run'",
            #     "alias Piranha='conda activate py27env && /mnt/data/liguowei/miniconda3/envs/py27env/bin/Piranha'",
            #     "alias piranha='conda activate py27env && /mnt/data/liguowei/miniconda3/envs/py27env/bin/Piranha'",
            #     "alias flye-minimap2='flye-minimap2 -t 32'",
            #     "alias flye='flye -t 32'",
            #     "alias Ribotaper=/root/data/liguowei/GenomeAgent/ribotaper/bin/Ribotaper.sh",
            #     "alias ribotaper='/root/data/liguowei/GenomeAgent/ribotaper/bin/Ribotaper.sh'",
            #     "alias plink='/mnt/data/liguowei/miniconda3/envs/GWAS/bin/plink'",
            #     "alias Piranha=/mnt/data/liguowei/miniconda3/bin/Piranha",
            #     "alias piranha=/mnt/data/liguowei/miniconda3/bin/Piranha",
            #     "### prefix end"
            #     ]:
            #     w.write(code + "\n")
            w.write(modify_content)
            # w.write("\n".join([
            #     "unset http_proxy",
            #     "unset https_proxy",]))
            # w.write("\n")

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

        elif self.running_env == 'eiflow':
            # 构建任务参数
            job_shell_path = self.job_shell + ".exe"
            create_job = eiflow_set_job(job_shell_path, eiflow_job_name, self.eiflow)
            job_id = eiflow_run_job(create_job)
            executor_info, execute_statu = eiflow_watch_job(eiflow_job_name, job_id, self.eiflow, self.max_stdout_line, self.max_stderr_line, self.max_log_len)
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

def check_and_create_conda_env(conda_path=None, env_name='bioenv', python_version='3.9', r_version='4.1.0', openjdk_version=None):
    if env_name == 'bioenv':
        python_version = '3.9'
        r_version = '4.1.0'
    elif env_name == 'Renv':
        python_version = '3.12'
        r_version = '4.3.3'
        openjdk_version = '17'

    # 获取现有的conda环境列表
    env_list = []
    try:
        if not conda_path:
            conda = subprocess.run(['/usr/bin/which', 'conda'], capture_output=True, text=True, check=True)
            conda_path = conda.stdout.strip()
        result = subprocess.run([conda_path, 'env', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        env_list = result.stdout + result.stderr
    except Exception as e:
        print_colored(e, 'RED')

    # 检查bioenv是否在环境列表中
    if env_name not in env_list:
        try:
            print_colored(f"conda env {env_name} do not exist, automatically create now...")
            # 创建新的conda环境
            if openjdk_version:
                res = subprocess.run([conda_path, 'create', '--name', env_name, f'conda-forge::python={python_version}', f'conda-forge::r-base={r_version}', f'conda-forge::openjdk={openjdk_version}', '-y'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                res = subprocess.run([conda_path, 'create', '--name', env_name, f'conda-forge::python={python_version}', f'conda-forge::r-base={r_version}', '-y'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return res.returncode == 0
        except EnvironmentError:
            return False
    else:
        return True

def check_and_create_conda(conda_path=None, conda_home='/home/health-user/miniconda/', url="https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh", filename="Miniconda3-py310_24.5.0-0-Linux-x86_64.sh"):
    install = False
    try:
        if not conda_path:
            conda = subprocess.run(['/usr/bin/which', 'conda'], capture_output=True, text=True, check=True)
            conda_path = conda.stdout.strip()
        install = subprocess.run([conda_path, '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if install.returncode == 0:
            install = True
    except Exception as e:
        print_colored(e, 'RED')
    if install:
        return install
    try:
        print_colored(f"download and install Miniconda: {url}")
        response = requests.get(url, stream=False)
        delete_file(filename)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        fd = os.open(filename, flags, modes)
        with os.fdopen(fd, "wb") as file:
            file.write(response.content)
        cmd = [
            ["/usr/bin/bash", filename, "-b", "-u", "-p", conda_home],
            ["source", f"{conda_home}bin/activate"],
            ["export", f"PATH={conda_home}bin:$PATH"],
            ["eval", f"$({conda_home}bin/conda shell.bash hook)"],
            ["export", f"PATH={conda_home}bin:$PATH"],
            [f"{conda_home}bin/conda", "init"],
            ["/usr/bin/echo", f"export PATH={conda_home}bin:$PATH", ">>", "~/.bashrc"],
            ["/usr/bin/echo", f"'eval $({conda_home}bin/conda shell.bash hook)'", ">>", "~/.bashrc"]
        ]
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        fd = os.open(filename + ".exe", flags, modes)
        with os.fdopen(fd, "w") as file:
            for i in cmd:
                file.write(" ".join(i) + "\n")
        res = subprocess.run(["/bin/bash", filename + ".exe"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0:
            install = True
            os.environ['PATH'] = f'{conda_home}bin:' + os.environ.get('PATH', '')
            conda_path = f'{conda_home}bin/conda'
    except subprocess.CalledProcessError:
        print_colored("Failed to install conda", "RED")
    delete_file(filename)
    delete_file(filename + ".exe")
    return install

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
    skip_content = ['libmamba Cache', 'conda info --envs', 'EnvironmentNameNotFound', "Using cache",
                    'Transaction', 'Prefix: ', 'All requested packages already installed', 'warning  libmamba',
                    'Checked', 'check zst',
                    '/linux-64', '/noarch', '─' * 10, 'B  conda-forge',
                    'The following NEW packages will be INSTALLED', 'The following packages will be downloaded',
                    'Collecting package metadatas', 'Solving environment', '## Package Plan ##',
                    'environment location', 'updated specs', 'was built under R version',
                    'Loading required package', 'Attaching package:', 'The following objects are masked', 'WARNING']
    res = []
    for i in content:
        if (not contains_any(i, skip_content) and
                '\n' != i and not bool(re.match(r'^\s*$|^[^a-zA-Z0-9]*$', i)) and
                sum([x.startswith("col") or x.startswith("row") for x in i.split(", ")]) < 3):
            res.append(f'[{type}] ' + i.strip())
    return res

def filter_stdout(content):
    content = re.sub(r'\b(col|row)\w+\b,?\s*', '', content)
    skip_content = ['libmamba Cache', 'conda info --envs', 'EnvironmentNameNotFound', "Using cache",
                    'Transaction', 'Prefix: ', 'All requested packages already installed', 'warning  libmamba',
                    'Checked', 'check zst',
                    '/linux-64', '/noarch', '─' * 10, 'B  conda-forge',
                    'The following NEW packages will be INSTALLED', 'The following packages will be downloaded',
                    'Collecting package metadatas', 'Solving environment', '## Package Plan ##',
                    'environment location', 'updated specs', 'was built under R version',
                    'Loading required package', 'Attaching package:', 'The following objects are masked', 'WARNING']
    for i in skip_content:
        if i in content:
            return True, content
    if '\n' == content or bool(re.match(r'^\s*$|^[^a-zA-Z0-9]*$', content)):
        return True, content
    if content.strip().endswith('Cached'):
        return True, content
    # if sum([x.startswith("col") or x.startswith("row") for x in content.split(", ")]) >= 3:
    #     return True, content
    return False, content

def filter_stderr(content):
    content = re.sub(r'\b(col|row)\w+\b,?\s*', '', content)
    skip_content = ['libmamba Cache', 'conda info --envs', 'EnvironmentNameNotFound', "Using cache",
                    'Transaction', 'Prefix: ', 'All requested packages already installed', 'warning  libmamba',
                    'Checked', 'check zst',
                    '/linux-64', '/noarch', '─' * 10, 'B  conda-forge',
                    'The following NEW packages will be INSTALLED', 'The following packages will be downloaded',
                    'Collecting package metadatas', 'Solving environment', '## Package Plan ##',
                    'environment location', 'updated specs', 'was built under R version',
                    'Loading required package', 'Attaching package:', 'The following objects are masked', 'WARNING']
    for i in skip_content:
        if i in content:
            return True, content
    if '\n' == content or bool(re.match(r'^\s*$|^[^a-zA-Z0-9]*$', content)):
        return True, content
    if content.strip().endswith('Cached'):
        return True, content
    # if sum([x.startswith("col") or x.startswith("row") for x in content.split(", ")]) >= 3:
    #     return True, content
    return False, content





def eiflow_set_job(job_shell_path, eiflow_job_name, eiflow):
    index = eiflow.obs_working_path.find(eiflow.project_name) + len(eiflow.project_name)
    job_workflow_input_sh = eiflow.project_name + ":/" + os.path.join(eiflow.obs_working_path[index:], job_shell_path).lstrip("/")
    job_workflow_input = f"{eiflow.workflow_task_name}.input-sh={job_workflow_input_sh};;"
    job_workflow_input += f"{eiflow.workflow_task_name}.input-dir={eiflow.obs_loading_path};;"
    job_workflow_input += f"{eiflow.workflow_task_name}.output-dir={eiflow.job_output_path}"

    create_job = [eiflow.health_tool, "create", "job", "-w", eiflow.workflow_id, "-n", eiflow_job_name, "-i", job_workflow_input]

    if 'nodeLabels' in eiflow.job_params.keys():
        create_job.append(f"-l {eiflow.job_params['nodeLabels']}")
    if 'priority' in eiflow.job_params.keys():
        create_job.append(f"-p {eiflow.job_params['priority']}")
    if 'io-acc-id' in eiflow.job_params.keys():
        create_job.append(f"-c {eiflow.job_params['io-acc-id']}")
    if 'io-acc-tasks' in eiflow.job_params.keys():
        create_job.append(f"-s {eiflow.job_params['io-acc-tasks']}")

    print_colored(eiflow.health_tool + " create " + " job " + " -w " + eiflow.workflow_id +
                  " -n " + eiflow_job_name + " -i \"" + job_workflow_input + "\"")


def eiflow_run_job(create_job: List) -> Tuple[bool, str]:
    result = None
    # 投递任务
    try:
        result = subprocess.run(create_job, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        print_colored(e, 'RED')
    return_code = result.returncode
    output = result.stdout
    errors = result.stderr

    # 检查任务投递情况
    job_id = None
    if return_code == 0:
        job_id = re.search(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
                           output).group(0)
        print_colored(f"[The job have been submitted with id: {job_id}!]", 'GREEN')
        return True, job_id
    else:
        # raise ValueError(f"The create job command execution failed; the return code: {return_code}; errors: {errors}")
        return False, (f"The create job command execution failed; \n"
                       f"the return code: {return_code}; \noutput: {output}; \nerrors: {errors}")


def eiflow_watch_job(eiflow_job_name, job_id, eiflow, max_stdout_line, max_stderr_line, max_log_len):
    # 远程任务，每隔1分钟，查询job状态
    status = eiflow_get_log_once(job_id, eiflow)
    print_colored("Check job status! " + " ".join([eiflow.health_tool, "get", "job", job_id, "--detail"]))
    total_time = 0
    while status in ["RUNNING", "PENDING"]:
        time.sleep(60)
        total_time += 60
        status = eiflow_get_log_once(job_id, eiflow)
        if total_time > 604800:
            print_colored(f"The job {eiflow_job_name} {job_id} has been running for over 7 days, please check it.", 'RED')
            raise ValueError(f"The job {eiflow_job_name} {job_id} has been running for over 7 days, please check it.")

    # 根据作业状态处理行为
    # RUNNING  CANCELLED FAILED SUCCEEDED
    # Succeeded、Running、Pending、Failed、Cancelling、Cancelled、Unknown
    # 当作业状态为成功或者失败时，获取日志
    if status == "SUCCEEDED" or status == "FAILED":
        time.sleep(60 * 5)
        log_info = None
        while not log_info:
            time.sleep(10)
            try:
                log_info = eiflow_get_log_once(job_id, eiflow, return_log=True)
            except Exception as e:
                time.sleep(3)
        # print(f"log_info:{}")
        _, _, _, executor_info = log_processor(log_info, max_stdout_line, max_stderr_line, max_log_len, path_prefix_to_remove=eiflow.obs_working_path, timestamp_length=20)
        if status == "SUCCEEDED":
            return executor_info, True
        if status == "FAILED":
            return executor_info, False
    # 通常来说，作业最终状态只为成功或者失败
    else:
        print_colored(f"The job status: {status}.", 'RED')
        print_colored(f"The job {eiflow_job_name} {job_id} status is abnormal, please check it.", 'RED')
        raise ValueError(f"The job status: {status}. The job {eiflow_job_name} {job_id} status is abnormal, please check it.")


def eiflow_get_job_once(job_id, eiflow):
    # 获取日志转储链接
    try:
        cmd = [eiflow.health_tool, "get", "job", job_id, "--detail"]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return_code = result.returncode
        output = json.loads(result.stdout)
        errors = result.stderr
        return result, return_code, output, errors
    except subprocess.CalledProcessError as e:
        cmd = ["health", "get", "job", job_id, "--detail"]
        print_colored("Failed to check job status! " + " ".join(cmd), 'RED')
        print_colored(e.stderr, 'RED')
    return None, None, None


def eiflow_get_log_once(job_id=None, eiflow=None, return_log=False):
    # 获取日志转储链接
    result, return_code, output, errors = eiflow_get_job_once(job_id, eiflow)

    if return_code == 0 and not return_log:
        return output["jobs"][0]["status"]

    elif return_code == 0 and return_log:
        if "log_storage_link" in output["jobs"][0]["task_runtime_info"][0]["sub_tasks"][0]:
            log_storage_link = output["jobs"][0]["task_runtime_info"][0]["sub_tasks"][0]["log_storage_link"]
            while not log_storage_link.startswith("https://eihealth-storage"):
                time.sleep(60)
                # 再次获取日志转储链接
                result, return_code, output, errors = eiflow_get_job_once(job_id, eiflow)
                log_storage_link = output["jobs"][0]["task_runtime_info"][0]["sub_tasks"][0]["log_storage_link"]
            url = log_storage_link.replace("\u0026", "&")
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Requests failed to get log information: status_code {response.status_code}")
            log_content = response.text.splitlines()
            return log_content
        else:
            print_colored("log_storage_link does not exist int job info", 'RED')
            raise ValueError

    else:
        print_colored(f"The job status command execution failed; the return code: {return_code}; errors: {errors}", 'RED')
        raise ValueError


Context = {
    "running_env": "local",  # 'local' or 'eiflow'
    "terminal_interactive": False,
    "eiflow": None,

    # llm setting
    "model": "",
    "base_url": None,
    "proxy": None,
    "proxy_verify": False,

    # data setting
    "task_config": {},
    "working_dir": "",
    "output_log_dir": "",
    "black_list": [
        "bioconductor-deseq2", "r-deseq2", "deseq2", "r-deseq2", "deseq2", "r", "r-base"
    ],

    # shell setting
    "conda_home": "/Users/liguowei/ubuntu/miniconda3/",
    "conda_bioenv": "bioenv",
    "conda_renv": "Renv",
    "do_execute": False,
    "debug": True,

    "current_agent": None,
    "current_input": None,

    # backend
    "last_step": None,
    "last_status": False,
    "task_history": [],
    "global_round": [0],
    "round_times": [0],
    "cur_round": 0,

    "global_message_index": [],
    "all_execute_code": [],
    "all_execute_log": [],
    "all_messages": [],
    "frontend_messages": [],
    "chat_history": [],

    # 变量配置：落地时的考虑，限制了任务的步骤重试次数等
    "max_round": 300,
    "max_times_per_round": 30,

    # 占位
    'rag_env': None,
    'runtime_tools': {},
    'runtime_agents': {},

    "activate_agent_state": {},

    "activate_tool_state": {
        "web_search": "deactivated",
        "visit_webpage": "deactivated",
        }
}

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
    if (bool(re.search(r'(^|[^a-zA-Z0-9])nano ', bash_code)) or
            bool(re.search(r'(^|[^a-zA-Z0-9])vi ', bash_code)) or
            bool(re.search(r'(^|[^a-zA-Z0-9])vim ', bash_code))):
        if (bool(re.search(r'-nano ', bash_code))
                or bool(re.search(r'-vi ', bash_code))
                or bool(re.search(r'-vim ', bash_code))):
            return True, None
        else:
            print_colored(
                f'Code Execute Statu: Failed! Using "nano"/"vi"/"vim" is not allowed in a non-interactive environment!',
                'ORANGE', debug
                )
            return False, f'Code Execute Statu: Failed! Using "nano"/"vi"/"vim" is not allowed in a non-interactive environment!'
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
    def __init__(self):
        self.debug = False
        self.black_list: List[str] = []
        self.max_times_per_round: int = 0
        self.eiflow = None

    def run(self, bash_code: str, context_variables) -> str:
        """
        This tool receive bash code in string format, then automatically invoke a Linux environment to execute the
        given code, and return a description of the execution status.
        """
        # init replace
        self.debug = context_variables['debug']
        self.black_list = context_variables['black_list']
        self.max_times_per_round = context_variables['max_times_per_round']
        self.eiflow = context_variables['eiflow']
        if self.eiflow is not None:
            self.shell_running_dir = self.eiflow.obs_working_path

        # record times for this round, failed when reaching limitation
        global_round = context_variables['global_round'][-1] + 1
        if len(context_variables['round_times']) <= global_round:
            context_variables['round_times'].append(0)
            assert len(context_variables['round_times']) == (global_round+1)

        # TODO 这里默认把脚本写到了f"{self.output_log_dir}/{self.global_round}.sh"这个位置
        # write original code script
        job_shell = os.path.join(context_variables['output_log_dir'], f"step{global_round}.sh" )
        if self.eiflow is not None:
            cmd_into_shell_dir = ""
        else:
            cmd_into_shell_dir = f"cd {context_variables['working_dir']}\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        fd = os.open(job_shell, flags, modes)
        with os.fdopen(fd, "w") as w:
            w.write(cmd_into_shell_dir + bash_code + "\n")

        # code rule check
        for func_name in Code_Rule_Functions.keys():
            statu, log = Code_Rule_Functions.get(func_name)(bash_code)
            if not statu:
                context_variables['last_status'] = False
                return log

        # record original code
        context_variables['all_execute_code'].append(bash_code)
        # print original code
        bash_code = re.sub(re.compile(r"```.*?\n", re.DOTALL), '', bash_code)
        bash_code = re.sub(re.compile(r"```", re.DOTALL), '', bash_code)
        print_colored(f"### code in {job_shell}\n```bash\n{bash_code}\n```", 'GREEN', self.debug)

        # config, modify, and write local code file
        if self.eiflow:
            runner = AgentCodeExecutor(running_env='eiflow',
                                       EIFlow=self.eiflow,
                                       conda_home=self.eiflow.conda_home,
                                       conda_bioenv=self.eiflow.conda_bioenv,
                                       conda_renv=self.eiflow.conda_renv,
                                       black_list=self.black_list)
        else:
            runner = AgentCodeExecutor(running_env=context_variables['running_env'],
                                       conda_home=context_variables['conda_home'],
                                       conda_bioenv=context_variables['conda_bioenv'],
                                       conda_renv=context_variables['conda_renv'],
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
        if context_variables['do_execute']:
            if self.eiflow:
                eiflow_job_name = self.eiflow.job_name + f"-step{global_round}"
                execute_log, execute_statu = runner.job_status_check(eiflow_job_name=eiflow_job_name)
            else:
                execute_log, execute_statu = runner.job_status_check()
                # print_colored(f"execute_log, execute_statu: {execute_statu}", "ORANGE")
        else:
            execute_log, execute_statu = f'Code Execute Statu: Success', True

        if 'Rscript' in bash_code:
            execute_log = execute_log.replace("[stderr] ", "").replace("[stdout] ", "")

        # record code result
        context_variables['all_execute_log'].append("\n```bash\n" + bash_code + "\n```\n" + execute_log.replace("[stdout] ", "").replace("[stderr] ", ""))

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
            context_variables['last_status'] = False
            context_variables['round_times'][global_round] += 1
            execute_log = re.sub(re.compile(r"```.*?\n", re.DOTALL), '', execute_log)
            execute_log = re.sub(re.compile(r"```", re.DOTALL), '', execute_log)
            print_colored(f'### Code Execute Statu: Failed {execute_statu}, log: \n```\n{execute_log}\n```\n', 'ORANGE', self.debug)
            return f'Code Execute Statu: Failed! The failed log: {execute_log}'

        # 成功 成功了不计数，因为工具可能会被并行跑
        context_variables['last_status'] = True
        context_variables['round_times'][global_round] += 0
        # 成功了记录当前步骤进队列
        context_variables['global_round'].append(global_round)
        # 初始化下一步骤的次数
        context_variables['round_times'].append(0)
        # 记录代码，和相关进度
        context_variables['task_history'].append(bash_code)
        context_variables['global_message_index'].append({
            'all_messages': len(context_variables['all_messages']),
            'frontend_messages': len(context_variables['frontend_messages']),
            'all_execute_code': len(context_variables['all_execute_code']),
            'all_execute_log': len(context_variables['all_execute_log'])})
        log_statu, blocked_log = block_files_in_log(bash_code, execute_log)
        if log_statu:
            print_colored(f'### Code Execute Statu: Success {execute_statu}! Log: \n```\n{blocked_log}\n```\n', 'ORANGE', self.debug)
            return f'Code Execute Statu: Success! Log: {blocked_log}'
        else:
            print_colored(f'### Code Execute Statu: Success {execute_statu}!', 'ORANGE', self.debug)
            return f'Code Execute Statu: Success!'

    def bash_code_llama_processor(self, bash_code, llm):
        system_prompt = ("You are a code debugger.\nYou must follow the rules: (1) make you response as simple as "
                         "possible; (2) try to keep the original bash code, only fix the error in codes; (3) do not "
                         "add other unnecessary codes.")
        prompt = "Check if any error in the bash codes and return your revised codes in block with ```bash  ``` :\n" + bash_code
        content = None
        while True:
            try:
                print(f"bash_code_run_tool llama_processor:")
                content = llm.call(prompt, system_prompt)
                if "```bash" in content.lower():
                    match = re.search(r"```bash(.*?)```", content, re.DOTALL)
                    content = match.group(1)
                else:
                    content = None
                print(f"bash_code_run_tool llama_processor:\n{content}")
            except:
                content = None
            if content:
                bash_code = content
                break
        return bash_code

# import getpass
# import os
#
# os.environ["BING_SUBSCRIPTION_KEY"] = "b073cd58ed56447fb0fc3e92a35e5e6e"
# os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
# from langchain_community.utilities import BingSearchAPIWrapper
#
# def web_search_tool(query: str) -> str:
#     """
#     This is a web search tool and enables safe, ad-free, location-aware search results, surfacing relevant information from billions of web documents.
#     """
#     search = BingSearchAPIWrapper(k=4)
#     result = search.run(query)
#     return result
#
