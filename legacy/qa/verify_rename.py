"""Verification of the HERC rename: the change must be identifier-only.

Four checks are performed:

1. **naming** — no pre-rename identifier survives outside the documents that
   deliberately list it;
2. **imports** — every intra-project import resolves and every imported name is
   defined in its target module (static, so torch is not required);
3. **defaults** — every configuration and loss default value is unchanged;
4. **layers** — every layer-constructor call has the same arguments, so the
   parameter budget (17,678) is preserved.

Usage:
    python verify_rename.py --old <path to the pre-rename tree> [--new <path>]

``--old`` must point at a copy of the code base before the rename; ``--new``
defaults to the repository this script lives in.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
import sys

# --------------------------------------------------------------------------- #
# identifier mapping
# --------------------------------------------------------------------------- #

RENAMED = {
    # modules
    'signals': 'cues',
    'failure_augmentation': 'error_augmentation',
    'fail2correct_losses': 'herc_loss',
    # classes
    'Fail2CorrectConfig': 'HERCConfig',
    'Fail2CorrectCell': 'HERCCell',
    'IterativeFail2Correct': 'HERCCorrector',
    'FailureDiagnosis3D': 'HERCDiagnosis',
    'FNRecoveryExpert3D': 'FalseNegativeRecoveryExpert',
    'FPSuppressionExpert3D': 'FalsePositiveSuppressionExpert',
    'ForegroundConfusionExpert3D': 'ForegroundConfusionExpert',
    'HierarchyConsistencyExpert3D': 'HierarchyCorrectionExpert',
    'FullResolutionBoundaryExpert3D': 'BoundaryCorrectionExpert',
    'Fail2CorrectCompositeLoss': 'HERCCompositeLoss',
    'SSDMoEFail2CorrectUNet': 'HERCUNet',
    'ConvNormAct3d': 'ConvNormAct',
    'ResidualStem3d': 'ResidualStem',
    'LightweightContextStem3d': 'ContextStem',
    'LightweightResidualOperator3d': 'ResidualOperator',
    # functions
    'uncertainty_signal': 'entropy_cue',
    'ambiguity_signal': 'ambiguity_cue',
    'spatial_gradient_magnitude': 'probability_gradient_cue',
    'image_gradient_magnitude': 'image_gradient_cue',
    'hierarchy_violation': 'hierarchy_violation_cue',
    'per_region_soft_boundary': 'soft_morphological_band',
    'build_failure_targets': 'build_route_targets',
    'build_region_failure_targets': 'build_region_route_targets',
    'build_atomic_failure_targets': 'build_atomic_route_targets',
    'induce_typed_failure': 'inject_typed_error',
    'induce_region_failure': 'inject_region_error',
    'induce_atomic_failure': 'inject_atomic_error',
    'atomic_fg_to_full': 'expand_foreground_to_full',
    'region_fail2correct_metrics': 'region_correction_metrics',
    'highest_res': 'highest_resolution',
    'replace_highest': 'replace_highest_resolution',
    'prepare_static': 'prepare_static_features',
    'region_target': '_resize_region_target',
    'atomic_target': '_resize_atomic_target',
    # fields, keys and arguments
    'feature_proj': 'feature_projection',
    'expert_residual_bound': 'semantic_residual_bound',
    'min_step_scale': 'min_step_size',
    'max_step_scale': 'max_step_size',
    'structure_expert': 'hierarchy_expert',
    'uncertainty': 'entropy',
    'prob_grad': 'prob_gradient',
    'pred_boundary': 'soft_band',
    'boundary_outward_raw': 'outward_raw',
    'boundary_inward_raw': 'inward_raw',
    'r_struct': 'r_h',
    'w_struct': 'w_h',
    'r_semantic_full': 'r_semantic',
    'r_boundary_full': 'r_boundary',
    'fail2correct_config': 'herc_config',
    'herc': 'herc',
    'set_fail2correct_enabled': 'set_herc_enabled',
    'set_fail2correct_trainable': 'set_herc_trainable',
    'fail2correct_enabled': 'herc_enabled',
    'use_synthetic_failure_warmup': 'use_synthetic_error_warmup',
    'synthetic_failure_target': 'synthetic_error_target',
    'synthetic_failure_type': 'synthetic_error_type',
    'f2c': 'herc',
    # local variables that were renamed for readability
    'frac': 'fraction',
    'still': 'still_active',
    'ff_proj': 'feature_proj',
    'pp': 'prob_full',
    'bb': 'band_full',
    'rr': 'route_full',
    'gg': 'grad',
    'boundary_input': 'band',
    'region_gate': 'band_gate',
    'feat_static': 'feature',
    'img': 'img',
    'prev_logits': 'previous_logits',
    'prev_aux': 'previous_aux',
    'tar': 'route_target',
    'kw': 'kw',
    'kwargs': 'kwargs',
}

FORBIDDEN = [
    'fail2correct', 'Fail2Correct', 'FAIL2CORRECT', 'f2c', 'F2C',
    'FailureDiagnosis3D', 'FNRecoveryExpert3D', 'FPSuppressionExpert3D',
    'HierarchyConsistencyExpert3D', 'FullResolutionBoundaryExpert3D',
    'ForegroundConfusionExpert3D', 'ConvNormAct3d', 'ResidualStem3d',
    'LightweightContextStem3d', 'LightweightResidualOperator3d',
    'IterativeFail2Correct', 'Fail2CorrectCell', 'Fail2CorrectConfig',
    'Fail2CorrectCompositeLoss', 'SSDMoEFail2CorrectUNet', 'SSDMoEFail2Correct',
    'structure_expert', 'expert_residual_bound', 'min_step_scale', 'max_step_scale',
    'induce_typed_failure', 'induce_region_failure', 'induce_atomic_failure',
    'build_failure_targets', 'uncertainty_signal', 'ambiguity_signal',
    'per_region_soft_boundary', 'spatial_gradient_magnitude', 'image_gradient_magnitude',
    'atomic_fg_to_full', 'region_fail2correct_metrics', 'r_struct', 'w_struct',
    'r_semantic_full', 'r_boundary_full',
]

#: Files that intentionally contain the pre-rename vocabulary.
DOC_ALLOWLIST = {'NAMING.md', 'README.md'}
#: The archived patch keeps its historical content and file name.
LEGACY_PREFIX = 'legacy/'

FILE_PAIRS = [
    ('umamba/nnunetv2/nets/fail2correct/blocks.py', 'umamba/nnunetv2/nets/herc/blocks.py'),
    ('umamba/nnunetv2/nets/fail2correct/diagnosis.py', 'umamba/nnunetv2/nets/herc/diagnosis.py'),
    ('umamba/nnunetv2/nets/fail2correct/experts.py', 'umamba/nnunetv2/nets/herc/experts.py'),
    ('umamba/nnunetv2/nets/fail2correct/cell.py', 'umamba/nnunetv2/nets/herc/cell.py'),
    ('umamba/nnunetv2/nets/fail2correct/config.py', 'umamba/nnunetv2/nets/herc/config.py'),
    ('umamba/nnunetv2/nets/fail2correct/signals.py', 'umamba/nnunetv2/nets/herc/cues.py'),
    ('umamba/nnunetv2/nets/fail2correct/targets.py', 'umamba/nnunetv2/nets/herc/targets.py'),
    ('umamba/nnunetv2/nets/fail2correct/failure_augmentation.py',
     'umamba/nnunetv2/nets/herc/error_augmentation.py'),
    ('umamba/nnunetv2/nets/SSDMoEFail2Correct.py', 'umamba/nnunetv2/nets/HERCUNet.py'),
    ('umamba/nnunetv2/training/loss/fail2correct_losses.py',
     'umamba/nnunetv2/training/loss/herc_loss.py'),
    ('umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerSSDMoEFail2Correct.py',
     'umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerHERC.py'),
]

CLASS_PAIRS = [
    ('Fail2CorrectConfig', 'HERCConfig'),
    ('FailureDiagnosis3D', 'HERCDiagnosis'),
    ('FNRecoveryExpert3D', 'FalseNegativeRecoveryExpert'),
    ('FPSuppressionExpert3D', 'FalsePositiveSuppressionExpert'),
    ('HierarchyConsistencyExpert3D', 'HierarchyCorrectionExpert'),
    ('FullResolutionBoundaryExpert3D', 'BoundaryCorrectionExpert'),
    ('ForegroundConfusionExpert3D', 'ForegroundConfusionExpert'),
    ('Fail2CorrectCompositeLoss', 'HERCCompositeLoss'),
    ('Fail2CorrectCell', 'HERCCell'),
    ('IterativeFail2Correct', 'HERCCorrector'),
]

TRACKED_LAYERS = {'Conv3d', 'GroupNorm', 'AdaptiveAvgPool3d', 'SiLU'}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def canonical(name: str) -> str:
    """Map an old identifier to its new name; new names map to themselves."""
    return RENAMED.get(name, name)


def defined_names(tree: ast.Module) -> set[str]:
    """Every name bound at module level."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split('.')[0] for alias in node.names
            )
    return names


def module_file(pkg_root: Path, parts: list[str]) -> Path | None:
    """Resolve a dotted module path to a file below ``pkg_root``."""
    base = pkg_root.joinpath(*parts)
    if base.with_suffix('.py').is_file():
        return base.with_suffix('.py')
    if (base / '__init__.py').is_file():
        return base / '__init__.py'
    return None


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #

def check_naming(new_root: Path) -> list[str]:
    problems = []
    for path in sorted(new_root.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        rel = str(path.relative_to(new_root)).replace('\\', '/')
        if rel in DOC_ALLOWLIST or rel.startswith(LEGACY_PREFIX):
            continue
        if path.suffix.lower() in {'.pyc', '.png', '.jpg', '.pdf', '.zip'}:
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        scrubbed = text.replace('legacy/FAIL2CORRECT_pre_rename.patch', '')
        scrubbed = scrubbed.replace('FAIL2CORRECT_pre_rename', '')
        for needle in FORBIDDEN:
            if needle in scrubbed:
                line = next(
                    (i for i, ln in enumerate(scrubbed.splitlines(), 1) if needle in ln), 0
                )
                problems.append(f"naming: '{needle}' survives in {rel}:{line}")
    return problems


def check_imports(new_root: Path) -> list[str]:
    problems = []
    pkg_root = new_root / 'umamba'
    trees: dict[Path, ast.Module] = {}
    for path in sorted(new_root.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        try:
            trees[path] = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except SyntaxError as exc:
            problems.append(f'syntax: {path.relative_to(new_root)}: {exc}')

    for path, tree in trees.items():
        try:
            pkg_parts = path.relative_to(pkg_root).with_suffix('').parts
        except ValueError:
            continue
        if pkg_parts[-1] == '__init__':
            pkg_parts = pkg_parts[:-1]

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                drop = node.level - (1 if path.name == '__init__.py' else 0)
                parts = list(pkg_parts[: len(pkg_parts) - drop])
                if node.module:
                    parts += node.module.split('.')
            elif node.module and node.module.split('.')[0] == 'nnunetv2':
                parts = node.module.split('.')
            else:
                continue

            target = module_file(pkg_root, parts)
            rel = path.relative_to(new_root)
            if target is None:
                problems.append(f"import: {rel}:{node.lineno} -> {'.'.join(parts)} unresolved")
                continue
            available = defined_names(trees[target]) if target in trees else set()
            for alias in node.names:
                if alias.name != '*' and alias.name not in available:
                    problems.append(
                        f"import: {rel}:{node.lineno} -> "
                        f"{'.'.join(parts)}.{alias.name} is not defined"
                    )
    return problems


def signature_defaults(path: Path, class_name: str) -> dict:
    """Literal defaults of a dataclass or of a class ``__init__``."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        defaults: dict = {}
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.value is not None:
                    defaults[item.target.id] = ast.literal_eval(item.value)
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                positional = item.args.posonlyargs + item.args.args
                slicing = positional[len(positional) - len(item.args.defaults):]
                for argument, default in zip(slicing, item.args.defaults):
                    defaults[argument.arg] = ast.literal_eval(default)
                for argument, default in zip(item.args.kwonlyargs, item.args.kw_defaults):
                    if default is not None:
                        defaults[argument.arg] = ast.literal_eval(default)
        return defaults
    raise KeyError(f'{class_name} not found in {path}')


def find_file(root: Path, class_name: str) -> Path:
    for path in root.rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        if f'class {class_name}' in path.read_text(encoding='utf-8', errors='ignore'):
            return path
    raise KeyError(class_name)


def check_defaults(old_root: Path, new_root: Path) -> list[str]:
    problems = []
    for old_name, new_name in CLASS_PAIRS:
        old_defaults = signature_defaults(find_file(old_root, old_name), old_name)
        new_defaults = signature_defaults(find_file(new_root, new_name), new_name)
        for key, value in old_defaults.items():
            mapped = canonical(key)
            if mapped not in new_defaults:
                problems.append(f'defaults: {new_name}.{mapped} disappeared (was {key})')
            elif new_defaults[mapped] != value:
                problems.append(
                    f'defaults: {new_name}.{mapped} changed {value!r} -> '
                    f'{new_defaults[mapped]!r}'
                )
    return problems


def constructor_calls(path: Path) -> Counter:
    """Multiset of tracked layer/component constructor calls."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    component_names = set(RENAMED.values())
    counter: Counter = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = canonical(ast.unparse(node.func))
        if name.split('.')[-1] not in TRACKED_LAYERS and name not in component_names:
            continue
        args = []
        for arg in node.args:
            try:
                args.append(repr(ast.literal_eval(arg)))
            except (ValueError, SyntaxError):
                args.append(f'<{type(arg).__name__}>')
        keywords = []
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                value = repr(ast.literal_eval(keyword.value))
            except (ValueError, SyntaxError):
                value = f'<{type(keyword.value).__name__}>'
            keywords.append((canonical(keyword.arg), value))
        counter[(name, tuple(args), tuple(sorted(keywords)))] += 1
    return counter


def check_layers(old_root: Path, new_root: Path) -> list[str]:
    problems = []
    for old_rel, new_rel in FILE_PAIRS:
        old_path, new_path = old_root / old_rel, new_root / new_rel
        if not old_path.is_file():
            problems.append(f'layers: missing {old_rel} in the old tree')
            continue
        only_old = constructor_calls(old_path) - constructor_calls(new_path)
        only_new = constructor_calls(new_path) - constructor_calls(old_path)
        for (name, args, keywords), count in only_old.items():
            problems.append(f'layers: {new_rel}: -{count}x {name}{args}{dict(keywords)}')
        for (name, args, keywords), count in only_new.items():
            problems.append(f'layers: {new_rel}: +{count}x {name}{args}{dict(keywords)}')
    return problems


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', required=True, type=Path,
                        help='pre-rename copy of the project')
    parser.add_argument('--new', type=Path,
                        default=Path(__file__).resolve().parents[2],
                        help='renamed project (default: this repository)')
    args = parser.parse_args()

    failures: list[str] = []
    for title, problems in (
        ('naming', check_naming(args.new)),
        ('imports', check_imports(args.new)),
        ('defaults', check_defaults(args.old, args.new)),
        ('layers', check_layers(args.old, args.new)),
    ):
        if problems:
            print(f'[{title}] {len(problems)} problem(s)')
            failures.extend(problems)
        else:
            print(f'[{title}] OK')

    if failures:
        print()
        for failure in failures:
            print('  ' + failure)
        return 1
    print('\nAll checks passed: the rename is identifier-only.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
