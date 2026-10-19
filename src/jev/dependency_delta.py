from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import xml.etree.ElementTree as ET

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None

from pydantic import BaseModel, ConfigDict, Field


class DepChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    ecosystem: str
    manifest: str
    section: str
    old_version: Optional[str] = None
    new_version: Optional[str] = None

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DepChange):
            return NotImplemented
        return (
            self.manifest,
            self.section,
            self.name,
            self.ecosystem,
            self.old_version or "",
            self.new_version or "",
        ) < (
            other.manifest,
            other.section,
            other.name,
            other.ecosystem,
            other.old_version or "",
            other.new_version or "",
        )


class DependencyDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    added: List[DepChange] = Field(default_factory=list)
    removed: List[DepChange] = Field(default_factory=list)
    changed: List[DepChange] = Field(default_factory=list)
    unparsed: List[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed or self.unparsed)

    def to_summary_text(self) -> str:
        lines: List[str] = []
        if self.added:
            items = [f"{c.name} ({c.new_version})" if c.new_version else c.name for c in self.added]
            lines.append(f"added: {', '.join(items)}")
        if self.removed:
            items = [f"{c.name} ({c.old_version})" if c.old_version else c.name for c in self.removed]
            lines.append(f"removed: {', '.join(items)}")
        if self.changed:
            items = [
                f"{c.name} ({c.old_version or 'none'} -> {c.new_version or 'none'})"
                for c in self.changed
            ]
            lines.append(f"changed: {', '.join(items)}")
        if self.unparsed:
            lines.append(f"unparsed: {', '.join(self.unparsed)}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> DependencyDelta:
        return cls.model_validate_json(json_str)



PEP508_REGEX = re.compile(
    r"^([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?)(?:\[[^\]]*\])?\s*(.*)$"
)

GRADLE_DEP_REGEX = re.compile(
    r"\b([a-zA-Z][a-zA-Z0-9_]*)\s*(?:\(\s*)?['\"]([^'\"\s:]+):([^'\"\s:]+):([^'\"\s:]+)['\"]\s*\)?"
)

KNOWN_UNSUPPORTED_BASENAMES = {
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "Gemfile",
    "composer.json",
    "build.sbt",
    "pubspec.yaml",
    "mix.exs",
    "Package.swift",
    "Podfile",
    "go.work",
    "libs.versions.toml",
    "deno.json",
    "deno.jsonc",
    "environment.yml",
    "environment.yaml",
    "Directory.Packages.props",
    "packages.config",
}

KNOWN_UNSUPPORTED_SUFFIXES = (
    ".csproj",
    ".fsproj",
    ".vbproj",
    ".gemspec",
    ".podspec",
)

LOCKFILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    "bun.lock",
    "poetry.lock",
    "uv.lock",
    "Cargo.lock",
    "go.sum",
    "Pipfile.lock",
    "Gemfile.lock",
    "composer.lock",
    "Podfile.lock",
}

MANIFEST_BASENAMES = {
    "package.json",
    "pyproject.toml",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
}


def _is_unsupported_manifest(basename: str) -> bool:
    if basename in KNOWN_UNSUPPORTED_BASENAMES:
        return True
    if basename.lower() in {b.lower() for b in KNOWN_UNSUPPORTED_BASENAMES}:
        return True
    for suffix in KNOWN_UNSUPPORTED_SUFFIXES:
        if basename.lower().endswith(suffix.lower()):
            return True
    return False


def _normalize_python_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_pep508_dep(entry: str) -> Tuple[str, Optional[str]]:
    clean = entry.split(";", 1)[0].split("#", 1)[0].strip()
    m = PEP508_REGEX.match(clean)
    if not m:
        return _normalize_python_name(clean), None
    raw_name = m.group(1)
    version = m.group(2).strip() or None
    return _normalize_python_name(raw_name), version


def _extract_version_from_val(val: Any) -> Optional[str]:
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        if "version" in val:
            return str(val["version"])
        if "git" in val:
            return str(val.get("rev") or val.get("tag") or val.get("branch") or val["git"])
        if "path" in val:
            return str(val["path"])
        return None
    return str(val) if val is not None else None


def _parse_package_json(text: str) -> Dict[Tuple[str, str], Optional[str]]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("package.json must be a JSON object")
    deps: Dict[Tuple[str, str], Optional[str]] = {}
    for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        sec_data = data.get(section)
        if isinstance(sec_data, dict):
            for name, ver in sec_data.items():
                deps[(section, str(name))] = str(ver) if ver is not None else None
    return deps


def _parse_pyproject_toml(text: str) -> Dict[Tuple[str, str], Optional[str]]:
    if tomllib is None:
        raise RuntimeError("tomllib not available")
    data = tomllib.loads(text)
    if not isinstance(data, dict):
        raise ValueError("pyproject.toml must be a TOML table")
    deps: Dict[Tuple[str, str], Optional[str]] = {}

    project = data.get("project")
    if isinstance(project, dict):
        proj_deps = project.get("dependencies")
        if isinstance(proj_deps, list):
            for entry in proj_deps:
                if isinstance(entry, str):
                    name, ver = _parse_pep508_dep(entry)
                    if name:
                        deps[("dependencies", name)] = ver

        opt_deps = project.get("optional-dependencies")
        if isinstance(opt_deps, dict):
            for group, group_list in opt_deps.items():
                if isinstance(group_list, list):
                    for entry in group_list:
                        if isinstance(entry, str):
                            name, ver = _parse_pep508_dep(entry)
                            if name:
                                deps[(f"optional-dependencies.{group}", name)] = ver

    tool = data.get("tool")
    if isinstance(tool, dict):
        poetry = tool.get("poetry")
        if isinstance(poetry, dict):
            p_deps = poetry.get("dependencies")
            if isinstance(p_deps, dict):
                for name, val in p_deps.items():
                    norm_name = _normalize_python_name(str(name))
                    if norm_name == "python":
                        continue
                    deps[("dependencies", norm_name)] = _extract_version_from_val(val)

            p_dev_deps = poetry.get("dev-dependencies")
            if isinstance(p_dev_deps, dict):
                for name, val in p_dev_deps.items():
                    norm_name = _normalize_python_name(str(name))
                    deps[("dev-dependencies", norm_name)] = _extract_version_from_val(val)

            p_group = poetry.get("group")
            if isinstance(p_group, dict):
                for group_name, group_data in p_group.items():
                    if isinstance(group_data, dict):
                        g_deps = group_data.get("dependencies")
                        if isinstance(g_deps, dict):
                            for name, val in g_deps.items():
                                norm_name = _normalize_python_name(str(name))
                                deps[(f"group.{group_name}.dependencies", norm_name)] = _extract_version_from_val(val)

    return deps


def _join_line_continuations(text: str) -> List[str]:
    lines = text.splitlines()
    joined_lines: List[str] = []
    current_line = ""
    for line in lines:
        stripped = line.strip()
        if current_line:
            current_line += " " + stripped
        else:
            current_line = stripped
        if current_line.endswith("\\"):
            current_line = current_line[:-1].rstrip()
        else:
            joined_lines.append(current_line)
            current_line = ""
    if current_line:
        joined_lines.append(current_line)
    return joined_lines


def _parse_requirements_txt(text: str) -> Dict[Tuple[str, str], Optional[str]]:
    deps: Dict[Tuple[str, str], Optional[str]] = {}
    for line in _join_line_continuations(text):
        if not line or line.startswith("#"):
            continue
        clean = line.split("#", 1)[0].strip()
        if not clean or clean.startswith("-") or "://" in clean:
            continue
        clean = clean.split(";", 1)[0].strip()
        clean = re.split(r"\s+--?[a-zA-Z0-9_-]+", clean)[0].strip()
        if not clean:
            continue
        name, ver = _parse_pep508_dep(clean)
        if name:
            deps[("dependencies", name)] = ver
    return deps


def _parse_go_mod(text: str) -> Dict[Tuple[str, str], Optional[str]]:
    deps: Dict[Tuple[str, str], Optional[str]] = {}
    in_require_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("require (") or stripped == "require (":
            in_require_block = True
            continue
        if in_require_block:
            if stripped == ")" or stripped.startswith(")"):
                in_require_block = False
                continue
            if "// indirect" in line:
                continue
            code = stripped.split("//", 1)[0].strip()
            parts = code.split()
            if len(parts) >= 2:
                name = parts[0]
                version = parts[1]
                deps[("require", name)] = version
        else:
            if stripped.startswith("require "):
                if "// indirect" in line:
                    continue
                code = stripped.split("//", 1)[0].strip()
                parts = code[len("require "):].strip().split()
                if len(parts) >= 2:
                    name = parts[0]
                    version = parts[1]
                    deps[("require", name)] = version
    return deps


def _parse_cargo_toml(text: str) -> Dict[Tuple[str, str], Optional[str]]:
    if tomllib is None:
        raise RuntimeError("tomllib not available")
    data = tomllib.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Cargo.toml must be a TOML table")
    deps: Dict[Tuple[str, str], Optional[str]] = {}

    for sec in ("dependencies", "dev-dependencies", "build-dependencies"):
        table = data.get(sec)
        if isinstance(table, dict):
            for name, val in table.items():
                deps[(sec, str(name))] = _extract_version_from_val(val)

    workspace = data.get("workspace")
    if isinstance(workspace, dict):
        w_deps = workspace.get("dependencies")
        if isinstance(w_deps, dict):
            for name, val in w_deps.items():
                deps[("workspace.dependencies", str(name))] = _extract_version_from_val(val)

    target = data.get("target")
    if isinstance(target, dict):
        for target_name, target_table in target.items():
            if isinstance(target_table, dict):
                for sec in ("dependencies", "dev-dependencies", "build-dependencies"):
                    table = target_table.get(sec)
                    if isinstance(table, dict):
                        for name, val in table.items():
                            deps[(f"target.{target_name}.{sec}", str(name))] = _extract_version_from_val(val)

    return deps


def _local_tag(elem: ET.Element) -> str:
    tag = elem.tag
    return tag.split("}", 1)[1] if "}" in tag else tag


def _parse_pom_xml(text: str) -> Dict[Tuple[str, str], Optional[str]]:
    root = ET.fromstring(text)
    deps: Dict[Tuple[str, str], Optional[str]] = {}

    def _traverse(elem: ET.Element, ancestors: List[str]):
        local = _local_tag(elem)
        current_ancestors = ancestors + [local]
        if local == "dependency":
            if "plugin" in ancestors:
                section = "plugin-dependencies"
            elif "dependencyManagement" in ancestors:
                section = "dependencyManagement"
            else:
                section = "dependencies"

            group_id = None
            artifact_id = None
            version = None
            for child in elem:
                ctag = _local_tag(child)
                if ctag == "groupId":
                    group_id = child.text.strip() if child.text else None
                elif ctag == "artifactId":
                    artifact_id = child.text.strip() if child.text else None
                elif ctag == "version":
                    version = child.text.strip() if child.text else None
            if group_id and artifact_id:
                name = f"{group_id}:{artifact_id}"
                deps[(section, name)] = version
            return

        for child in elem:
            _traverse(child, current_ancestors)

    _traverse(root, [])
    return deps


def _parse_build_gradle(text: str) -> Dict[Tuple[str, str], Optional[str]]:
    deps: Dict[Tuple[str, str], Optional[str]] = {}
    no_block_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    for line in no_block_comments.splitlines():
        line = line.split("//", 1)[0].strip()
        if not line:
            continue
        for m in GRADLE_DEP_REGEX.finditer(line):
            sec = m.group(1)
            group = m.group(2)
            artifact = m.group(3)
            ver = m.group(4)
            deps[(sec, f"{group}:{artifact}")] = ver
    return deps


def _get_manifest_parser_and_ecosystem(
    basename: str,
) -> Optional[Tuple[Callable[[str], Dict[Tuple[str, str], Optional[str]]], str]]:
    lower = basename.lower()
    if lower == "package.json":
        return _parse_package_json, "js_ts"
    if lower == "pyproject.toml":
        return _parse_pyproject_toml, "python"
    if lower.startswith("requirements") and lower.endswith(".txt"):
        return _parse_requirements_txt, "python"
    if lower == "go.mod":
        return _parse_go_mod, "go"
    if lower == "cargo.toml":
        return _parse_cargo_toml, "rust"
    if lower == "pom.xml":
        return _parse_pom_xml, "maven"
    if lower in ("build.gradle", "build.gradle.kts"):
        return _parse_build_gradle, "gradle"
    return None


def compute_dependency_delta(
    changes: Mapping[str, Tuple[Optional[str], Optional[str]]],
) -> DependencyDelta:
    added: List[DepChange] = []
    removed: List[DepChange] = []
    changed: List[DepChange] = []
    unparsed: List[str] = []

    for path, (old_text, new_text) in changes.items():
        basename = re.split(r"[\\/]", path)[-1]
        if basename in LOCKFILES:
            continue
        if _is_unsupported_manifest(basename):
            unparsed.append(path)
            continue

        parser_info = _get_manifest_parser_and_ecosystem(basename)
        if not parser_info:
            continue

        parser, ecosystem = parser_info

        old_deps: Dict[Tuple[str, str], Optional[str]] = {}
        new_deps: Dict[Tuple[str, str], Optional[str]] = {}
        failed = False

        if old_text is not None:
            try:
                old_deps = parser(old_text)
            except Exception:
                failed = True

        if not failed and new_text is not None:
            try:
                new_deps = parser(new_text)
            except Exception:
                failed = True

        if failed:
            unparsed.append(path)
            continue

        if old_text is None and new_text is None:
            continue

        all_keys = set(old_deps.keys()) | set(new_deps.keys())
        for sec, name in all_keys:
            if (sec, name) in new_deps and (sec, name) not in old_deps:
                added.append(
                    DepChange(
                        name=name,
                        ecosystem=ecosystem,
                        manifest=path,
                        section=sec,
                        old_version=None,
                        new_version=new_deps[(sec, name)],
                    )
                )
            elif (sec, name) in old_deps and (sec, name) not in new_deps:
                removed.append(
                    DepChange(
                        name=name,
                        ecosystem=ecosystem,
                        manifest=path,
                        section=sec,
                        old_version=old_deps[(sec, name)],
                        new_version=None,
                    )
                )
            else:
                old_v = old_deps[(sec, name)]
                new_v = new_deps[(sec, name)]
                if old_v != new_v:
                    changed.append(
                        DepChange(
                            name=name,
                            ecosystem=ecosystem,
                            manifest=path,
                            section=sec,
                            old_version=old_v,
                            new_version=new_v,
                        )
                    )

    return DependencyDelta(
        added=sorted(added),
        removed=sorted(removed),
        changed=sorted(changed),
        unparsed=sorted(list(set(unparsed))),
    )


def is_dependency_manifest_path(path: str) -> bool:
    basename = re.split(r"[\\/]", path)[-1]
    if basename in LOCKFILES:
        return False
    if _is_unsupported_manifest(basename):
        return True
    if _get_manifest_parser_and_ecosystem(basename) is not None:
        return True
    return False


def format_dependency_summary(delta_json: Optional[str]) -> str:
    if delta_json is None:
        return "Dependencies: not computed"
    try:
        delta = DependencyDelta.from_json(delta_json)
    except Exception:
        return "Dependencies: not computed"
    if delta.is_empty and not delta.unparsed:
        return "Dependencies: no changes"
    summary_text = delta.to_summary_text()
    return f"Dependencies: {summary_text}"

