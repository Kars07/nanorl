from __future__ import annotations

from collections.abc import Iterable

import verifiers.v1 as vf

from tiny_repo_repair_v1.models import E2BToolConfig, RepairState, TinyRepoRepairData
from tiny_repo_repair_v1.servers.tool import E2BRepoToolset


def _case(category: str, prompt: str, source: str, assertion: str) -> dict[str, object]:
    public_test = "from solution import target\n\ndef test_smoke():\n    assert callable(target)\n"
    checker = f"python -c \"from solution import target; {assertion}\""
    return {
        "category": category,
        "prompt": (
            f"{prompt}\n\nThe implementation is /home/user/workspace/solution.py and the public "
            "smoke test is /home/user/workspace/test_public.py. Read solution.py first, edit "
            "only solution.py, run pytest, and call e2b_submit. Do not invent filenames. "
            "All shell and file operations must use the e2b tools."
        ),
        "system_prompt": (
            "You are a repository repair agent. Use the provided e2b tools immediately. "
            "Make the smallest implementation fix and always finish by calling e2b_submit."
        ),
        "files": {"solution.py": source, "test_public.py": public_test},
        "check_command": checker,
    }


CASES = [
    _case("solved-add", "The implementation is already correct. Verify it adds two integers and submit without unnecessary edits.", "def target(a, b):\n    return a + b\n", "assert target(7, 5) == 12; assert target(-2, 2) == 0"),
    _case("solved-range", "The implementation is already correct. Verify its inclusive integer range and submit without unnecessary edits.", "def target(start, end):\n    return list(range(start, end + 1))\n", "assert target(2, 5) == [2, 3, 4, 5]"),
    _case("solved-parser", "The implementation is already correct. Verify whitespace-tolerant integer parsing and submit without unnecessary edits.", "def target(text):\n    return int(text)\n", "assert target(' 42 ') == 42"),
    _case("solved-validation", "The implementation is already correct. Verify negative-age validation and submit without unnecessary edits.", "def target(age):\n    if age < 0:\n        raise ValueError('negative age')\n    return age\n", "assert target(3) == 3; assert_raises = False\ntry:\n target(-1)\nexcept ValueError:\n assert_raises = True\nassert assert_raises"),
    _case("wrong-return", "Fix target so it adds two integers.", "def target(a, b):\n    return a - b\n", "assert target(7, 5) == 12; assert target(-2, 2) == 0"),
    _case("off-by-one", "Fix target so it returns the inclusive integer range.", "def target(start, end):\n    return list(range(start, end))\n", "assert target(2, 5) == [2, 3, 4, 5]"),
    _case("parser", "Fix target so surrounding whitespace is accepted when parsing an integer.", "def target(text):\n    if text != text.strip():\n        raise ValueError('spaces')\n    return int(text)\n", "assert target(' 42 ') == 42"),
    _case("validation", "Fix target to reject negative ages with ValueError.", "def target(age):\n    return age\n", "assert target(3) == 3; assert_raises = False\ntry:\n target(-1)\nexcept ValueError:\n assert_raises = True\nassert assert_raises"),
    _case("cli-option", "Fix target to recognize --verbose in an argv list.", "def target(argv):\n    return '-v' in argv\n", "assert target(['--verbose']); assert not target([])"),
    _case("path", "Fix target to return only the final component on POSIX or Windows paths.", "def target(path):\n    return path.split('/')[-1]\n", "assert target(r'a\\b\\c.txt') == 'c.txt'; assert target('/a/b.txt') == 'b.txt'"),
    _case("mutation", "Fix target so it returns a sorted copy without mutating the input.", "def target(values):\n    values.sort()\n    return values\n", "x=[3,1,2]; assert target(x)==[1,2,3]; assert x==[3,1,2]"),
    _case("exception", "Fix target so division by zero returns None while other divisions work.", "def target(a, b):\n    return a / b\n", "assert target(8,2)==4; assert target(1,0) is None"),
    _case("dedupe", "Fix target to remove duplicates while preserving first-seen order.", "def target(values):\n    return list(set(values))\n", "assert target([3,1,3,2,1]) == [3,1,2]"),
    _case("casefold", "Fix target to compare text case-insensitively for Unicode.", "def target(a, b):\n    return a.lower() == b.lower()\n", "assert target('Straße','STRASSE')"),
    _case("empty-input", "Fix target so an empty sequence returns None instead of raising.", "def target(values):\n    return values[0]\n", "assert target([]) is None; assert target([9]) == 9"),
    _case("default-arg", "Fix target so calls do not share mutable default state.", "def target(value, items=[]):\n    items.append(value)\n    return items\n", "assert target(1)==[1]; assert target(2)==[2]"),
    _case("boolean", "Fix target so only positive even integers return True.", "def target(n):\n    return n % 2 == 0\n", "assert target(2); assert not target(0); assert not target(-2); assert not target(3)"),
    _case("boundary", "Fix target to clamp a number to the inclusive [low, high] interval.", "def target(value, low, high):\n    return min(low, max(high, value))\n", "assert target(-1,0,10)==0; assert target(5,0,10)==5; assert target(20,0,10)==10"),
    _case("unicode", "Fix target to count Unicode characters rather than UTF-8 bytes.", "def target(text):\n    return len(text.encode('utf-8'))\n", "assert target('café') == 4; assert target('🙂') == 1"),
    _case("rounding", "Fix target to round Decimal monetary values to exactly two places.", "from decimal import Decimal\ndef target(value):\n    return round(float(value), 2)\n", "assert str(target(Decimal('2.675'))) == '2.68'; assert str(target(Decimal('1.005'))) == '1.01'"),
    _case("iterator", "Fix target so it sums any iterable without consuming it twice.", "def target(values):\n    if not list(values):\n        return 0\n    return sum(values)\n", "assert target(iter([1,2,3])) == 6; assert target(iter([])) == 0"),
    _case("dictionary", "Fix target to invert a dictionary and group duplicate values.", "def target(mapping):\n    return {v:k for k,v in mapping.items()}\n", "assert target({'a':1,'b':1,'c':2}) == {1:['a','b'],2:['c']}"),
    _case("substring", "Fix target to count overlapping substring occurrences.", "def target(text, needle):\n    return text.count(needle)\n", "assert target('aaaa','aa') == 3; assert target('abc','x') == 0"),
    _case("normalization", "Fix target to flatten one level of nested lists while preserving order.", "def target(groups):\n    return sorted(sum(groups, []))\n", "assert target([[3,1],[2],[1]]) == [3,1,2,1]"),
]


def _build_dataset(size: int = 150) -> list[dict[str, object]]:
    """Create deterministic repo instances across the audited repair archetypes."""
    templates = CASES
    dataset: list[dict[str, object]] = []
    for idx in range(size):
        template = templates[idx % len(templates)]
        files = dict(template["files"])
        files["solution.py"] = f"# dataset_instance={idx}\n{files['solution.py']}"
        dataset.append(
            {
                **template,
                "category": f"{template['category']}-instance-{idx:03d}",
                "prompt": f"Dataset instance {idx:03d}. {template['prompt']}",
                "files": files,
            }
        )
    return dataset


CASES = _build_dataset()


class TinyRepoRepairTaskConfig(vf.TaskConfig):
    e2b: E2BToolConfig = E2BToolConfig()


class TinyRepoRepairTask(vf.Task[TinyRepoRepairData, RepairState, TinyRepoRepairTaskConfig]):
    @classmethod
    def toolsets(cls, config: TinyRepoRepairTaskConfig) -> list[vf.Toolset]:
        return [E2BRepoToolset(config.e2b)]

    async def validate(self, runtime: vf.Runtime) -> bool:
        compile(self.data.files["solution.py"], "solution.py", "exec")
        return bool(self.data.check_command and self.data.prompt)

    @vf.reward(weight=1.0)
    async def repository_passed(self, trace: vf.Trace) -> float:
        return trace.state.score

    @vf.metric
    async def submitted(self, trace: vf.Trace) -> float:
        return float(trace.state.submitted)

    @vf.metric
    async def sandbox_commands(self, trace: vf.Trace) -> float:
        return float(trace.state.command_count)


class TinyRepoRepairConfig(vf.TasksetConfig):
    num_tasks: int = 150
    start: int = 0
    task: TinyRepoRepairTaskConfig = TinyRepoRepairTaskConfig()


class TinyRepoRepairTaskset(vf.Taskset[TinyRepoRepairTask, TinyRepoRepairConfig]):
    def load(self) -> Iterable[TinyRepoRepairTask]:
        stop = self.config.start + self.config.num_tasks
        for idx, case in enumerate(CASES[self.config.start : stop], self.config.start):
            yield TinyRepoRepairTask(TinyRepoRepairData(idx=idx, **case), self.config.task)
