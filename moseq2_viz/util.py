"""
General utility functions fors loading and organizing data.
"""

import re
import os
import json
import hashlib
import h5py
import numpy as np
from glob import glob
from typing import Union
import ruamel.yaml as yaml
from cytoolz import curry, compose
from cytoolz.curried import valmap
from cytoolz.dicttoolz import dissoc, assoc
from cytoolz.itertoolz import first, groupby
from os.path import join, exists, dirname, splitext


def camel_to_snake(s):
    """
    Convert CamelCase to snake_case

    Args:
    s (str): string to convert to snake case

    Returns:
    (str): snake_case string
    """
    # https://gist.github.com/jaytaylor/3660565
    _underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
    _underscorer2 = re.compile(r"([a-z0-9])([A-Z])")

    subbed = _underscorer1.sub(r"\1_\2", s)
    return _underscorer2.sub(r"\1_\2", subbed).lower()


def get_index_hits(config_data, metadata, key, v):
    """
    Search for matching keys in given index file metadata dict and return a list of booleans indicating that a session was found.

    Args:
    config_data (dict): dictionary containing boolean search filters [lowercase, negative]
    metadata (list): list of session metadata dict objects
    key (str): metadata key being searched for
    v (str): value of the corresponding key to be found

    Returns:
    hits (list): list of booleans indicating the found sessions to be updated in add_group_wrapper()
    """

    if config_data["lowercase"] and config_data["negative"]:
        # Convert keys to lowercase and return inverse selection
        hits = [re.search(v, meta[key].lower()) is None for meta in metadata]
    elif config_data["lowercase"]:
        # Convert keys to lowercase
        hits = [re.search(v, meta[key].lower()) is not None for meta in metadata]
    elif config_data["negative"]:
        # Return inverse selection
        hits = [re.search(v, meta[key]) is None for meta in metadata]
    else:
        # Default search
        hits = [re.search(v, meta[key]) is not None for meta in metadata]

    return hits


def clean_dict(dct):
    """
    Cast numpy.array into lists and np.generic data into scalar values.

    Args:
    dct (dict): dictionary with values to clean.

    Returns:
    (dict): dictionary with standardized value types.
    """

    def clean_entry(e):
        if isinstance(e, dict):
            out = clean_dict(e)
        elif isinstance(e, np.ndarray):
            out = e.tolist()
        elif isinstance(e, np.generic):
            out = np.asscalar(e)
        else:
            out = e
        return out

    return valmap(clean_entry, dct)


def _load_h5_to_dict(file: h5py.File, path: str) -> dict:
    """
    Load h5 contents to dictionary.

    Args:
    file (h5py.File): open h5py File object.
    path (str): path within h5 to dict to load.

    Returns:
    ans (dict): loaded dictionary from h5 dataset or group
    """

    ans = {}
    if isinstance(file[path], h5py.Dataset):
        # only use the final path key to add to `ans`
        ans[path.split("/")[-1]] = file[path][()]
    else:
        for key, item in file[path].items():
            if isinstance(item, h5py.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py.Group):
                ans[key] = _load_h5_to_dict(file, "/".join([path, key]))
    return ans


def h5_to_dict(h5file, path: str = "/") -> dict:
    """
    Load h5 dict contents to a dict variable.

    Args:
    h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file. Default: /

    Returns:
    out (dict): dictionary of all h5 contents
    """

    if isinstance(h5file, str):
        with h5py.File(h5file, "r") as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, (h5py.File, h5py.Group)):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception("File input not understood. Use an h5 file path or file handle")
    return out


def get_timestamps_from_h5(h5file: str) -> np.ndarray:
    """
    Return a dict of timestamps from h5file.

    Args:
    h5file (str): path to h5 file.

    Returns:
    (np.ndarray): timestamps from extraction within the h5file.
    """

    with h5py.File(h5file, "r") as f:
        # v0.1.3 or greater data format
        is_new = "timestamps" in f
    if is_new:
        return h5_to_dict(h5file, "timestamps")["timestamps"]
    else:
        return h5_to_dict(h5file, "metadata/timestamps")["timestamps"]


def get_metadata_path(h5file):
    """
    Return path within h5 file that contains the kinect extraction metadata.

    Args:
    h5file (str): path to h5 file.

    Returns:
    (str): path to acquistion metadata within h5 file.
    """

    with h5py.File(h5file, "r") as f:
        if "/metadata/acquisition" in f:
            return "/metadata/acquisition"
        elif "/metadata/extraction" in f:
            return "/metadata/extraction"
        else:
            raise KeyError("acquisition metadata not found")


def load_changepoint_distribution(cpfile):
    """
    Load changepoint durations from given model free changepoints file.

    Args:
    cpfile (str): Path to changepoints h5 file.

    Returns:
    (numpy.array): Array of changepoint durations.
    """

    cps = h5_to_dict(cpfile, "cps")
    cp_dist = map(compose(np.diff, np.squeeze), cps.values())
    return np.concatenate(list(cp_dist))


def load_timestamps(timestamp_file, col=0):
    """
    Read timestamps from space delimited text file.

    Args:
    timestamp_file (str): path to timestamp file
    col (int): column to load.

    Returns:
    ts (np.ndarray): loaded array of timestamps
    """

    ts = np.loadtxt(timestamp_file, delimiter=" ")
    if ts.ndim > 1:
        return ts[:, col]
    elif col > 0:
        raise Exception(
            f"Timestamp file {timestamp_file} does not have more than one column of data"
        )
    return ts


def parse_index(index: Union[str, dict]) -> tuple:
    """
    Load an index file, and use extraction UUIDs to sort index.

    Args:
    index (str or dict): if str, must be a path to the index file. If dict, must be the unsorted index.

    Returns:
    index (dict): loaded index file contents in a dictionary
    sorted_index (dict): index where the files have been sorted by UUID and pca_score path.
    """

    from_dict = True
    if isinstance(index, str):
        index_dir = dirname(index)
        index = read_yaml(index)
        from_dict = False

    files = index["files"]

    sorted_index = groupby("uuid", files)
    # grab first entry in list, which is a dict
    sorted_index = valmap(first, sorted_index)
    # remove redundant uuid entry
    sorted_index = valmap(lambda d: dissoc(d, "uuid"), sorted_index)
    # tuple-ize the path entry, join with the index file dirname
    if not from_dict:
        sorted_index = valmap(
            lambda d: assoc(d, "path", tuple(join(index_dir, x) for x in d["path"])),
            sorted_index,
        )
    else:
        sorted_index = valmap(
            lambda d: assoc(d, "path", tuple(d["path"])), sorted_index
        )

    if index["pca_path"] is None:
        raise Exception("Please add pac_score path to moseq2-index.yaml")
    uuid_sorted = {
        "files": sorted_index,
        "pca_path": (
            join(index_dir, index["pca_path"]) if not from_dict else index["pca_path"]
        ),
    }

    return index, uuid_sorted


def get_sorted_index(index_file: str) -> dict:
    """
    Return the sorted index from an index_file path.

    Args:
    index_file (str): path to index file.

    Returns:
    sorted_ind (dict): dictionary of loaded sorted index file contents
    """

    _, sorted_ind = parse_index(index_file)
    return sorted_ind


def h5_filepath_from_sorted(sorted_index_entry: dict) -> str:
    """
    Get the h5 extraction file path from a sorted index entry

    Args:
    sorted_index_entry (dict): get filepath from sorted index.

    Returns:
    (str): a str containing the extraction filepath
    """

    return first(sorted_index_entry["path"])


def recursive_find_h5s(root_dir=None, ext=".h5", yaml_string="{}.yaml"):
    """
    Recursively find h5 files, along with yaml files with the same basename.

    Args:
    root_dir (str): path to directory containing h5, if None, searches the current working directory
    ext (str): extension to search for.
    yaml_string (str): yaml file format name.

    Returns:
    h5s (list): list of paths to h5 files
    dicts (list): list of paths to metadata files
    yamls (list): list of paths to yaml files
    """

    if root_dir is None:
        root_dir = os.getcwd()

    def has_frames(h5f):
        """Checks if the supplied h5 file has a frames key"""
        with h5py.File(h5f, "r") as f:
            return "frames" in f

    def h5_to_yaml(h5f):
        return yaml_string.format(splitext(h5f)[0])

    # make function to test if yaml file with same basename as h5 file exists
    yaml_exists = compose(exists, h5_to_yaml)

    # grab all files with ext = .h5
    files = glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True)
    # keep h5s that have a yaml file associated with them
    to_keep = filter(yaml_exists, files)
    # keep h5s that have a frames key
    to_keep = filter(has_frames, to_keep)

    h5s = list(to_keep)
    yamls = list(map(h5_to_yaml, h5s))
    dicts = list(map(read_yaml, yamls))

    return h5s, dicts, yamls


def read_yaml(yaml_path: str):
    """
    Read a given yaml file path into a dictionary object.

    Args:
    yaml_path (str): path to yaml file to read.

    Returns:
    loaded (dict): loaded yaml file contents.

    """
    with open(yaml_path, "r") as f:
        loaded = yaml.safe_load(f)
    return loaded


# from https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    Slice a numpy.array into subarrays with a given stride

    Args:
    a (numpy.array): array to get subarrays from.
    L (int): window length.
    S (int): stride size.

    Returns:
    (np.ndarray): sliced subarrays
    """

    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


@curry
def star(f, args):
    """
    Apply a function to a tuple of args, by expanding the tuple into each of the function's parameters.

    Args:
    f (function): a function that takes multiple arguments
    args (tuple): : a tuple to expand into ``f``

    Returns:
    the output of ``f``
    """

    return f(*args)


def assert_model_and_index_uuids_match(model, index):
    """
    Assert that both the model and index file contain the same set of UUIDs.

    Args:
    model (str or dict): str path to the model or dictionary of model.
    index (str or dict): str path to index file (moseq2-index.yaml) or dictionary of index data.
    """
    from moseq2_viz.model.util import parse_model_results

    if isinstance(model, str) and exists(model):
        # Load the model
        model = parse_model_results(model)
    if isinstance(index, str) and exists(index):
        # Read index file
        index = get_sorted_index(index)

    index_uuids = set(index["files"])
    model_uuids = set(model["metadata"]["uuids"])

    assert index_uuids == model_uuids, "Index file UUIDS must match the model UUIDs."


class ProvenanceError(ValueError):
    """
    Raised when provenance is corrupt or files about to be pooled are
    incompatible.

    Subclasses ValueError so existing ``except ValueError`` handlers keep
    working, but is catchable on its own for callers that want to offer an
    explicit override for compatibility conflicts. Corrupt records are never
    overrideable.
    """


# Extract writes the first path. The second is a historical fallback. Every
# reader uses this order and rejects conflicting dual records.
PIPELINE_PROVENANCE_PATHS = (
    'metadata/extraction/pipeline',
    'metadata/pipeline',
)

FLIP_RECORD_PATHS = (
    'metadata/processing/flip_classifier',
    'metadata/processing/flip_classifier_journal',
)
FLIP_RECORD_UPDATE_ATTR = 'flip_classifier_record_update_in_progress'
LEGACY_FLIP_ATTR = 'flip_classifier_applied'

# Scalars that must not enter analysis. Kept in this low-level module so public
# plotting, model, and scalar helpers can share one guard without circular
# imports.
DEPRECATED_SCALARS = {
    'velocity_3d_px': (
        "velocity_3d_px mixed units: its x/y terms are pixels-per-frame while "
        "its z term (from height_ave_mm) is millimetres-per-frame, so the value "
        "has no coherent physical unit. Use 'velocity_3d_mm' for 3D speed (all "
        "axes in mm), 'velocity_2d_mm' for horizontal speed in mm, or "
        "'velocity_2d_px' for horizontal speed in pixels."
    ),
}


def check_scalar_is_usable(scalar):
    """Raise when a requested scalar is deprecated and invalid for analysis."""
    requested = [scalar] if isinstance(scalar, str) else list(scalar)
    for name in requested:
        if name in DEPRECATED_SCALARS:
            raise ValueError(DEPRECATED_SCALARS[name])


def validate_scalar_dataframe(scalar_df, scalar_names=None):
    """Validate explicit scalar names or every column in a dataframe."""
    names = scalar_df.columns if scalar_names is None else scalar_names
    check_scalar_is_usable(names)


def _json_mapping_from_h5(f, dataset, label):
    """Read a JSON mapping or raise a non-overrideable provenance error."""
    try:
        raw = f[dataset][()]
        if isinstance(raw, bytes):
            raw = raw.decode()
        record = json.loads(raw)
    except (OSError, UnicodeError, TypeError, ValueError, KeyError) as error:
        raise ProvenanceError(
            "CORRUPT {} at {!r}: {}: {}".format(
                label, dataset, type(error).__name__, error
            )
        )
    if not isinstance(record, dict):
        raise ProvenanceError(
            "CORRUPT {} at {!r}: JSON must be an object/mapping, got {}".format(
                label, dataset, type(record).__name__
            )
        )
    return record


def _provenance_candidates(h5_path):
    """Return the one documented provenance-path order plus a custom path."""
    candidates = list(PIPELINE_PROVENANCE_PATHS)
    if h5_path and h5_path not in candidates:
        candidates.insert(0, h5_path)
    return candidates


def _read_extraction_provenance(f, h5_path):
    """Return ABSENT as None, VALID as a mapping, and raise on CORRUPT."""
    records = []
    for candidate in _provenance_candidates(h5_path):
        if candidate in f:
            records.append((
                candidate,
                _json_mapping_from_h5(f, candidate, 'pipeline provenance'),
            ))
    if not records:
        return None

    canonical = {
        json.dumps(record, sort_keys=True, separators=(',', ':'))
        for _, record in records
    }
    if len(canonical) > 1:
        raise ProvenanceError(
            'CORRUPT pipeline provenance: conflicting records exist at {}. '
            'Readers must not silently choose one.'.format(
                ', '.join(repr(path) for path, _ in records)
            )
        )
    return records[0][1]


def _read_flip_correction_policy(f):
    """
    Derive the app flip-correction policy without modifying extraction records.

    Incomplete, failed, corrupt, legacy-only, and interrupted states are
    ambiguous for orientation data and therefore hard-fail pooling.
    """
    if f.attrs.get(FLIP_RECORD_UPDATE_ATTR, False):
        raise ProvenanceError(
            'Flip-correction provenance is ambiguous: the processing-record '
            'update sentinel is still set.'
        )

    records = []
    for candidate in FLIP_RECORD_PATHS:
        if candidate not in f:
            continue
        record = _json_mapping_from_h5(
            f, candidate, 'flip-correction processing record'
        )
        try:
            sequence = int(record.get('_journal_sequence', 0))
        except (TypeError, ValueError) as error:
            raise ProvenanceError(
                "CORRUPT flip-correction processing record at {!r}: {}".format(
                    candidate, error
                )
            )
        records.append((sequence, candidate, record))

    if not records:
        if f.attrs.get(LEGACY_FLIP_ATTR, False):
            raise ProvenanceError(
                'Flip-correction provenance is ambiguous: only the legacy '
                'Boolean latch is present, so the classifier and settings are '
                'unknown.'
            )
        return 'not_applied'

    max_sequence = max(sequence for sequence, _, _ in records)
    latest = [
        (path, record)
        for sequence, path, record in records
        if sequence == max_sequence
    ]
    if len(latest) > 1:
        canonical = {
            json.dumps(record, sort_keys=True, separators=(',', ':'))
            for _, record in latest
        }
        if len(canonical) > 1:
            raise ProvenanceError(
                'Flip-correction provenance is ambiguous: record slots have '
                'the same journal sequence but conflicting contents.'
            )
    _, record = latest[0]
    state = record.get('state')
    if state != 'complete':
        raise ProvenanceError(
            'Flip-correction provenance is not poolable: latest processing '
            'state is {!r}, not complete.'.format(state)
        )

    compatibility_fields = (
        'package',
        'version',
        'git_sha',
        'classifier_sha256',
        'chunk_size',
        'chunk_overlap',
        'smoothing',
        'frame_path',
    )
    missing = [
        field for field in compatibility_fields
        if record.get(field) in (None, '')
    ]
    if missing:
        raise ProvenanceError(
            'Flip-correction provenance is ambiguous: complete record is '
            'missing compatibility field(s) {}.'.format(
                ', '.join(repr(field) for field in missing)
            )
        )
    compatibility = {
        field: record[field]
        for field in compatibility_fields
    }
    compatibility['state'] = 'complete'
    return json.dumps(compatibility, sort_keys=True, separators=(',', ':'))


def read_flip_correction_policy(path):
    """Return the effective app flip-correction policy for one extraction."""
    try:
        with h5py.File(path, 'r') as f:
            return _read_flip_correction_policy(f)
    except ProvenanceError:
        raise
    except OSError as error:
        raise ProvenanceError(
            'CORRUPT/unreadable extraction {!r}: {}: {}'.format(
                path, type(error).__name__, error
            )
        )


def read_pipeline_provenance(path, h5_path='metadata/pipeline'):
    """
    Read the provenance record written by moseq2-extract / moseq2-pca into an h5.

    Args:
    path (str): path to an h5 file (extraction results or pca scores/changepoints).
    h5_path (str): dataset path holding the JSON provenance record.

    Returns:
    provenance (dict): the VALID extraction record augmented in memory with
        the effective flip-correction policy, or None when the record is ABSENT.

    Raises:
    ProvenanceError: when a present record is CORRUPT, when both candidate
        paths conflict, or when flip-correction state is ambiguous.
    """
    try:
        with h5py.File(path, 'r') as f:
            record = _read_extraction_provenance(f, h5_path)
            flip_policy = _read_flip_correction_policy(f)
    except ProvenanceError:
        raise
    except OSError as error:
        raise ProvenanceError(
            'CORRUPT/unreadable extraction {!r}: {}: {}'.format(
                path, type(error).__name__, error
            )
        )
    if record is None:
        return None
    effective = dict(record)
    policies = dict(effective.get('policies', {}))
    policies['flip_correction'] = flip_policy
    effective['policies'] = policies
    return effective


def describe_provenance(provenance):
    """Render a provenance record as a short human-readable string."""
    if not provenance:
        return ('UNKNOWN provenance (no usable record; generation history '
                'cannot be inferred)')
    policies = ', '.join(f'{k}={v}' for k, v in sorted(provenance.get('policies', {}).items()))
    return (f"{provenance.get('package', '?')} {provenance.get('version', '?')} "
            f"@ {str(provenance.get('git_sha', '?'))[:10]} "
            f"({provenance.get('written', '?')}) [{policies}]")


#: Output policies that change the numbers in an extraction's scalars. Two
#: extractions pooled into one analysis must agree on all of these. The names
#: mirror moseq2_extract.util.EXTRACT_OUTPUT_POLICIES; they are duplicated here
#: rather than imported so moseq2-viz does not take a runtime dependency on
#: moseq2-extract just to read a list of strings.
EXTRACTION_POOLING_POLICIES = (
    'scalar_px_to_mm',
    'area_units',
    'background_dropout',
    'roi_plane_fit',
    'hampel_filter',
    'depth_range_detection',
    'velocity_3d_px',
)

SCALAR_POOLING_POLICIES = EXTRACTION_POOLING_POLICIES + ('flip_correction',)


def _provenance_policy_conflicts(records):
    """
    Find output policies that disagree across a set of provenance records.

    Args:
    records (dict): mapping of file path -> provenance record (never None).

    Returns:
    conflicts (dict): mapping of policy key -> {policy value: [paths]}, for
        every key whose value is not identical across all records. A key that
        is missing from some records is a conflict too; the absent value is
        reported as the string '<absent>'.
    """
    conflicts = {}
    all_keys = set()
    for record in records.values():
        all_keys.update((record or {}).get('policies', {}).keys())

    for key in sorted(all_keys):
        by_value = {}
        for path, record in records.items():
            value = (record or {}).get('policies', {}).get(key, '<absent>')
            by_value.setdefault(str(value), []).append(path)
        if len(by_value) > 1:
            conflicts[key] = by_value
    return conflicts


def _format_provenance_conflicts(conflicts, records):
    """Render a policy-conflict report that names the keys and the files."""
    lines = []
    for key, by_value in conflicts.items():
        lines.append(f"  policy '{key}' disagrees:")
        for value, paths in sorted(by_value.items()):
            lines.append(f"    {value!r}:")
            for path in sorted(paths):
                lines.append(f"      {path}")
    lines.append('  per-file provenance:')
    for path, record in sorted(records.items()):
        lines.append(f'    {path}: {describe_provenance(record)}')
    return '\n'.join(lines)


def enforce_consistent_provenance(paths, h5_path='metadata/pipeline',
                                  allow_mixed=False, context='this analysis',
                                  required_policies=None, records=None):
    """
    Refuse to pool files that were produced by incompatible pipeline versions.

    It is meant to be called at a pooling boundary -- the point where per-session
    files stop being separate and start being one dataset -- because that is
    where a version mismatch stops being visible and starts being a silently
    wrong number.

    The rules, from most to least severe:

    * **Conflicting policies -> error.** Two stamped files declare different
      values for the same output policy (e.g. one was extracted with the
      small-angle focal length and one with the pinhole projection). Their
      millimetre scalars are on different scales; pooling them is never correct.
    * **Stamped mixed with unstamped -> error.** An unstamped file has UNKNOWN
      provenance, so its policies cannot be assumed to match stamped records.
    * **All unstamped -> warning.** Every file has UNKNOWN provenance. Pooling
      is allowed, but the pipeline cannot establish compatibility.
    * **Different packages -> error.** Package-specific policy maps do not prove
      cross-package lineage or compatibility.
    * **All stamped and agreeing -> silent.**

    Comparison is on the semantic policy tags only. Two files built from
    different commits agree as long as their declared policies agree, so a
    documentation-only commit never invalidates otherwise compatible data.

    Args:
    paths (list): h5 paths about to be combined. Used to label the records and,
        unless ``records`` is given, to read them.
    h5_path (str): dataset path holding the JSON provenance record.
    allow_mixed (bool): opt out of the hard errors. The mismatch is still
        reported in full, as a loud warning, so it cannot pass unnoticed.
    context (str): short phrase naming what the files are being pooled into,
        used in the message (e.g. 'the scalar dataframe').
    required_policies (iterable): policy keys that must be established for this
        kind of pooling (e.g. :data:`SCALAR_POOLING_POLICIES`). A key that no
        stamped file declares cannot be compared, so it is reported as a
        warning; a key some files declare and others do not is a conflict and
        is handled by the normal rules.
    records (dict): optional pre-read ``path -> record`` mapping. When given,
        no files are opened and effective app processing state must already be
        represented in each record's policy map.

    Returns:
    consistent (bool): True if every file is stamped and all policies agree.

    Raises:
    ProvenanceError: on conflicting policies, or on a stamped/unstamped mix,
        unless ``allow_mixed`` is set.
    """
    import warnings

    paths = list(paths)
    if not paths:
        return True

    flip_policies = None
    if records is None:
        # Read flip state even when extraction provenance is ABSENT. Otherwise
        # two unstamped files could silently mix app-corrected and uncorrected
        # angle semantics in the all-UNKNOWN warning branch.
        flip_policies = {p: read_flip_correction_policy(p) for p in paths}
        records = {p: read_pipeline_provenance(p, h5_path) for p in paths}
    unstamped = sorted(p for p, r in records.items() if r is None)
    stamped = {p: r for p, r in records.items() if r is not None}

    def _report(message):
        if allow_mixed:
            warnings.warn(
                message
                + '\n\nallow_mixed=True was set, so this is being allowed. The '
                  'numbers below are being combined despite being known to be '
                  'incompatible. Results should not be trusted until the '
                  'affected sessions are reprocessed with one pipeline version.')
            return False
        raise ProvenanceError(
            message
            + '\n\nReprocess the affected sessions with a single pipeline '
              'version so they agree, or pass allow_mixed=True to override '
              'this check (the mismatch will still be reported as a warning).')

    if flip_policies is not None:
        by_policy = {}
        for path, policy in flip_policies.items():
            by_policy.setdefault(policy, []).append(path)
        if len(by_policy) > 1:
            details = []
            for policy, policy_paths in sorted(by_policy.items()):
                label = (
                    'not_applied'
                    if policy == 'not_applied'
                    else 'complete correction ({})'.format(
                        hashlib.sha256(policy.encode()).hexdigest()[:12]
                    )
                )
                details.append(
                    '  {}:\n{}'.format(
                        label,
                        '\n'.join('    ' + path for path in sorted(policy_paths)),
                    )
                )
            return _report(
                'Refusing to pool files with different flip-correction '
                'processing policies into {}.\n{}'.format(
                    context, '\n'.join(details)
                )
            )

    if stamped and unstamped:
        return _report(
            f'Refusing to pool files with mixed provenance into {context}.\n'
            f'{len(stamped)} of {len(paths)} files carry a pipeline provenance '
            f'record and {len(unstamped)} do not. Unstamped files have UNKNOWN '
            'provenance; their generation history and output policies cannot '
            'be inferred or assumed to match.\n'
            '  unstamped:\n'
            + '\n'.join(f'    {p}' for p in unstamped)
            + '\n  stamped:\n'
            + '\n'.join(f'    {p}: {describe_provenance(r)}'
                        for p, r in sorted(stamped.items())))

    if not stamped:
        warnings.warn(
            f'None of the {len(paths)} files being pooled into {context} carry '
            'a usable pipeline provenance record; provenance is UNKNOWN for '
            'every file, so compatibility cannot be established. Reprocess '
            'them to get verifiable records.')
        return False

    packages = sorted({(record or {}).get('package', '?')
                       for record in stamped.values()})
    if len(packages) > 1:
        return _report(
            f'Refusing to claim cross-package provenance compatibility for '
            f'{context}. The records come from {packages}, whose policy maps '
            'are package-specific and do not share a lineage identifier. '
            'Comparing one record from each package would be a no-op, not a '
            'compatibility check.')

    conflicts = _provenance_policy_conflicts(stamped)
    if conflicts:
        keys = ', '.join(repr(k) for k in conflicts)
        return _report(
            f'Refusing to pool files with conflicting output policies into '
            f'{context}.\nThe following policies disagree across the '
            f'{len(stamped)} files: {keys}.\n'
            'Files that disagree on a policy were produced by code with '
            'different numerical behaviour, so their scalars are on different '
            'scales or definitions.\n'
            + _format_provenance_conflicts(conflicts, stamped))

    # Every file agrees. Separately check that the policies this kind of pooling
    # depends on were actually established: a key no stamped file declares was
    # not compared, so agreement on it has not been shown, only assumed.
    if required_policies:
        declared = set()
        for record in stamped.values():
            declared.update((record or {}).get('policies', {}).keys())
        undeclared = [k for k in required_policies if k not in declared]
        if undeclared:
            warnings.warn(
                f'The files being pooled into {context} agree on every policy '
                'they declare, but none of them declares '
                + ', '.join(repr(k) for k in undeclared)
                + '. Those behaviours could not be compared, so compatibility '
                  'is established only for the policies that are recorded.')
            return False

    return True
