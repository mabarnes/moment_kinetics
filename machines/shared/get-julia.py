#!/usr/bin/env python3

import json
from pathlib import Path
from sys import exit
from urllib.request import urlopen, urlretrieve


# Lightweight wrapper for version numbers, to avoid adding any depedencies. Should work
# for Julia's version numbers at least
class SemVer:
    def __init__(self, version_str):
        self.version_str = version_str
        temp = version_str.split("-")
        if len(temp) > 2:
            raise ValueError(
                f"Expected at most one '-' in `version_str`, got {version_str}"
            )
        elif len(temp) == 2:
            just_version_str = temp[0]
            suffix = temp[1]
        else:
            just_version_str = temp[0]
            suffix = None

        self.is_stable = suffix is None
        self.version = tuple(int(x) for x in just_version_str.split("."))

    # Define this to make SemVer sortable
    def __lt__(self, other):
        if self.version == other.version:
            return self.is_stable < other.is_stable
        else:
            return self.version < other.version

    def __str__(self):
        return self.version_str


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Download Julia into this directory"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Specify a version. By default, get the latest stable version",
    )
    parser.add_argument(
        "--source", type=bool, default=False, help="Get source code rather than binary?"
    )
    parser.add_argument(
        "--os", type=str, default="linux", help="Operating system to download for"
    )
    parser.add_argument(
        "--arch", type=str, default="x86_64", help="Architecture to download for"
    )
    parser.add_argument(
        "--musl", type=bool, default=False, help="On Linux, get MUSL version?"
    )
    args = parser.parse_args()

    version_info = json.loads(
        urlopen("https://julialang-s3.julialang.org/bin/versions.json").read()
    )

    if args.version is None:
        julia_versions = [SemVer(x) for x in version_info.keys()]

        stable_versions = [x for x in julia_versions if x.is_stable]

        # Get lastest stable version
        version = str(max(stable_versions))
    else:
        version = args.version

    possible_versions = version_info[version]["files"]
    if args.source:
        # Get source-code download
        url = f"https://github.com/JuliaLang/julia/releases/download/v{version}/julia-{version}.tar.gz"
    else:
        # Get binary for platform given by args.os and args.arch
        version = [
            x
            for x in possible_versions
            if x["os"] == args.os and x["arch"] == args.arch
        ]

        if args.os == "linux":
            # Probably never want the 'musl' version, but allow for it with an argument
            if args.musl:
                version = [x for x in version if "musl" in x["triplet"]]
            else:
                version = [x for x in version if "musl" not in x["triplet"]]

        if len(version) > 1:
            raise ValueError(
                f"Expected a single possibility for os={args.os} and arch={args.arch} with musl={args.musl}. Got {len(version)}."
            )
        else:
            version = version[0]
        url = version["url"]

    tarfile_name = url.split("/")[-1]
    output_path = Path(args.output_dir, tarfile_name)
    urlretrieve(url, output_path)
    print(tarfile_name)
    exit(0)
