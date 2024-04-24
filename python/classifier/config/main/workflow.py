import shlex
from collections import defaultdict

import fsspec
import yaml
from classifier.task import ArgParser, EntryPoint, main
from classifier.task.parse._dict import _mapping_scheme, mapping
from classifier.utils import YamlIndentSequence
from yaml.representer import Representer

_MAX_WIDTH = 10


def _merge_args(opts: list[str]) -> list[str]:
    if (len(opts) <= 1) or (not opts[0].startswith(main._DASH)):
        return opts
    else:
        merged = []
        current = [opts[0]]
        for opt in opts[1:]:
            if (len(opt) > _MAX_WIDTH) or (opt != shlex.quote(opt)):
                if len(current) > 0:
                    merged.append(shlex.join(current))
                    current.clear()
                merged.append(opt)
            else:
                current.append(opt)
        if len(current) > 0:
            merged.append(shlex.join(current))
        return merged


def _parse_opts(mod: str, opts: list[str]):
    output = {main._MODULE: mod}
    merged = []
    group = []
    for opt in opts:
        scheme = _mapping_scheme(opt)[0]
        if (scheme is not None) or (opt.startswith(main._DASH)):
            merged.extend(_merge_args(group))
            group.clear()
        if scheme is not None:
            merged.append(mapping(opt))
        else:
            group.append(opt)
    merged.extend(_merge_args(group))
    if len(merged) > 0:
        output[main._OPTION] = merged
    return output


class Main(main.Main):
    _no_monitor = True
    _no_load = True

    argparser = ArgParser(
        prog="workflow",
        description="Generate a workflow file from command line arguments",
        workflow=[
            ("main", f"generate workflow"),
        ],
    )
    argparser.add_argument(
        "workflow",
        help="output path to workflow file",
    )
    argparser.add_argument(
        "main",
        help="main task",
    )

    def run(self, parser: EntryPoint):
        from base_class.system.eos import EOS

        output = EOS(self.opts.workflow)
        workflow = defaultdict(list)
        workflow[main._MAIN] = _parse_opts(
            self.opts.main, parser.args[main._MAIN][1][2:]
        )
        for k in parser._keys:
            for mod, opts in parser.args[k]:
                workflow[k].append(_parse_opts(mod, opts))

        yaml.add_representer(defaultdict, Representer.represent_dict)
        with fsspec.open(output, "wt") as f:
            yaml.dump(workflow, f, sort_keys=False, Dumper=YamlIndentSequence)
