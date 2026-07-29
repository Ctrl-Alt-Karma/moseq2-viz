"""
Tests for the pipeline-provenance gate that guards the pooling boundary.

The failure mode these cover is the half-reprocessed dataset: some sessions
extracted with one version of the pipeline, some with another, silently
concatenated into a single dataframe and then compared between groups. The
numbers look fine; they are simply on different scales.
"""

import json
import shutil
import tempfile
import warnings
from os.path import join
from unittest import TestCase

import h5py

from moseq2_viz.util import (SCALAR_POOLING_POLICIES, ProvenanceError,
                             describe_provenance,
                             enforce_consistent_provenance,
                             provenance_from_model,
                             read_pipeline_provenance)


#: a full, self-consistent extraction policy map, matching the key set in
#: moseq2_extract.util.EXTRACT_OUTPUT_POLICIES
FULL_POLICIES = {
    'scalar_px_to_mm': 'pinhole',
    'area_units': 'mm2-xy-product',
    'background_dropout': 'masked',
    'roi_plane_fit': 'lstsq-inlier-refit',
    'hampel_filter': 'mad-threshold-both-axes',
    'depth_range_detection': 'histogram-mode',
    'velocity_3d_px': 'invalid-mixed-units-omitted',
}


def _write_h5(path, provenance=None, h5_path='metadata/pipeline'):
    """Create a minimal h5 with (or without) a provenance record."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('scalars/centroid_x_px', data=[1.0, 2.0, 3.0])
        if provenance is not None:
            f.create_dataset(h5_path, data=json.dumps(provenance))
    return path


def _record(**policies):
    return {
        'package': 'moseq2-extract',
        'version': '1.2.0',
        'git_sha': 'abcdef0123456789',
        'written': '2026-07-29T00:00:00',
        'policies': policies,
    }


class TestProvenanceGate(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, name):
        return join(self.tmp, name)

    def test_all_stamped_and_agreeing_is_silent(self):
        paths = [_write_h5(self._path(f'ok{i}.h5'),
                           _record(focal_length='pinhole', velocity_3d_px='omitted'))
                 for i in range(3)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertTrue(enforce_consistent_provenance(paths))
        self.assertEqual([str(w.message) for w in caught], [])

    def test_conflicting_policy_raises_and_names_key_and_files(self):
        good = _write_h5(self._path('pinhole.h5'), _record(focal_length='pinhole'))
        bad = _write_h5(self._path('smallangle.h5'), _record(focal_length='small-angle'))

        with self.assertRaises(ProvenanceError) as ctx:
            enforce_consistent_provenance([good, bad])

        message = str(ctx.exception)
        # the offending policy key must be named
        self.assertIn("'focal_length'", message)
        # both conflicting values must be named
        self.assertIn('pinhole', message)
        self.assertIn('small-angle', message)
        # both files must be named, so the user knows which to reprocess
        self.assertIn(good, message)
        self.assertIn(bad, message)

    def test_policy_present_in_one_file_only_is_a_conflict(self):
        old = _write_h5(self._path('old.h5'), _record(focal_length='pinhole'))
        new = _write_h5(self._path('new.h5'),
                        _record(focal_length='pinhole', velocity_3d_px='omitted'))

        with self.assertRaises(ProvenanceError) as ctx:
            enforce_consistent_provenance([old, new])

        message = str(ctx.exception)
        self.assertIn("'velocity_3d_px'", message)
        self.assertIn('<absent>', message)
        # focal_length agrees, so it must NOT be reported as a conflict
        self.assertNotIn("policy 'focal_length' disagrees", message)

    def test_stamped_mixed_with_unstamped_raises(self):
        stamped = _write_h5(self._path('stamped.h5'), _record(focal_length='pinhole'))
        legacy = _write_h5(self._path('legacy.h5'), None)

        with self.assertRaises(ProvenanceError) as ctx:
            enforce_consistent_provenance([stamped, legacy])

        message = str(ctx.exception)
        self.assertIn(stamped, message)
        self.assertIn(legacy, message)
        self.assertIn('unstamped', message)

    def test_all_unstamped_warns_but_does_not_raise(self):
        paths = [_write_h5(self._path(f'legacy{i}.h5'), None) for i in range(3)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertFalse(enforce_consistent_provenance(paths))
        self.assertEqual(len(caught), 1)
        self.assertIn('provenance is UNKNOWN', str(caught[0].message))

    def test_allow_mixed_downgrades_conflict_to_a_loud_warning(self):
        good = _write_h5(self._path('pinhole.h5'), _record(focal_length='pinhole'))
        bad = _write_h5(self._path('smallangle.h5'), _record(focal_length='small-angle'))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertFalse(enforce_consistent_provenance([good, bad], allow_mixed=True))

        self.assertEqual(len(caught), 1)
        message = str(caught[0].message)
        # the override must not make the report quieter: same detail, plus a
        # statement that the result is untrustworthy
        self.assertIn("'focal_length'", message)
        self.assertIn(good, message)
        self.assertIn(bad, message)
        self.assertIn('should not be trusted', message)

    def test_allow_mixed_downgrades_stamped_unstamped_mix(self):
        stamped = _write_h5(self._path('stamped.h5'), _record(focal_length='pinhole'))
        legacy = _write_h5(self._path('legacy.h5'), None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertFalse(
                enforce_consistent_provenance([stamped, legacy], allow_mixed=True))

        self.assertEqual(len(caught), 1)
        self.assertIn('should not be trusted', str(caught[0].message))

    def test_empty_input_is_consistent(self):
        self.assertTrue(enforce_consistent_provenance([]))

    def test_single_stamped_file_is_consistent(self):
        path = _write_h5(self._path('only.h5'), _record(focal_length='pinhole'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertTrue(enforce_consistent_provenance([path]))
        self.assertEqual([str(w.message) for w in caught], [])

    def test_error_message_offers_the_override(self):
        good = _write_h5(self._path('a.h5'), _record(focal_length='pinhole'))
        bad = _write_h5(self._path('b.h5'), _record(focal_length='small-angle'))
        with self.assertRaises(ProvenanceError) as ctx:
            enforce_consistent_provenance([good, bad])
        self.assertIn('allow_mixed=True', str(ctx.exception))

    def test_provenance_error_is_a_value_error(self):
        self.assertTrue(issubclass(ProvenanceError, ValueError))


class TestRequiredScalarPolicies(TestCase):
    """
    The policies that must be established before extraction scalars are pooled.
    Each of these changes the numbers, so agreement on it is load-bearing.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pair(self, policies_a, policies_b):
        a = _write_h5(join(self.tmp, 'a.h5'), _record(**policies_a))
        b = _write_h5(join(self.tmp, 'b.h5'), _record(**policies_b))
        return a, b

    def test_the_required_key_set_is_what_extract_declares(self):
        self.assertEqual(set(SCALAR_POOLING_POLICIES), set(FULL_POLICIES))

    def test_identical_full_policies_pass_silently(self):
        a, b = self._pair(FULL_POLICIES, FULL_POLICIES)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertTrue(enforce_consistent_provenance(
                [a, b], required_policies=SCALAR_POOLING_POLICIES))
        self.assertEqual([str(w.message) for w in caught], [])

    def test_every_required_policy_is_actually_compared(self):
        """Flip one policy at a time; each must be caught on its own."""
        for key in SCALAR_POOLING_POLICIES:
            other = dict(FULL_POLICIES)
            other[key] = 'something-else'
            a, b = self._pair(FULL_POLICIES, other)
            with self.assertRaises(ProvenanceError, msg=f'{key} was not compared') as ctx:
                enforce_consistent_provenance(
                    [a, b], required_policies=SCALAR_POOLING_POLICIES)
            message = str(ctx.exception)
            self.assertIn(repr(key), message)
            self.assertIn(str(FULL_POLICIES[key]), message)
            self.assertIn('something-else', message)
            self.assertIn(a, message)
            self.assertIn(b, message)

    def test_pinhole_plus_small_angle_fails(self):
        other = dict(FULL_POLICIES, scalar_px_to_mm='small-angle')
        a, b = self._pair(FULL_POLICIES, other)
        with self.assertRaises(ProvenanceError) as ctx:
            enforce_consistent_provenance([a, b],
                                          required_policies=SCALAR_POOLING_POLICIES)
        self.assertIn('pinhole', str(ctx.exception))
        self.assertIn('small-angle', str(ctx.exception))

    def test_different_area_policies_fail(self):
        other = dict(FULL_POLICIES, area_units='linear-mm-per-px')
        a, b = self._pair(FULL_POLICIES, other)
        with self.assertRaises(ProvenanceError) as ctx:
            enforce_consistent_provenance([a, b],
                                          required_policies=SCALAR_POOLING_POLICIES)
        self.assertIn("'area_units'", str(ctx.exception))

    def test_agreeing_but_incomplete_policies_warn(self):
        """
        Files that agree on everything they declare, but never declare a
        required policy, have not been shown to be compatible on it.
        """
        partial = {'scalar_px_to_mm': 'pinhole'}
        a, b = self._pair(partial, partial)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertFalse(enforce_consistent_provenance(
                [a, b], required_policies=SCALAR_POOLING_POLICIES))
        message = str(caught[0].message)
        self.assertIn("'area_units'", message)
        self.assertNotIn("'scalar_px_to_mm'", message)

    def test_different_git_sha_with_same_policies_is_fine(self):
        """A documentation-only commit must not invalidate compatible data."""
        a = _write_h5(join(self.tmp, 'a.h5'),
                      dict(_record(**FULL_POLICIES), git_sha='1111111111'))
        b = _write_h5(join(self.tmp, 'b.h5'),
                      dict(_record(**FULL_POLICIES), git_sha='2222222222',
                           version='1.3.0', written='2026-08-01T00:00:00'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertTrue(enforce_consistent_provenance(
                [a, b], required_policies=SCALAR_POOLING_POLICIES))
        self.assertEqual([str(w.message) for w in caught], [])


class TestCrossPackageScope(TestCase):
    """
    Package-specific policy maps do not establish PCA/model shared lineage.
    A single record from each package must not pass as a compatibility check.
    """

    def test_disjoint_packages_are_not_claimed_compatible(self):
        pca = {'package': 'moseq2-pca', 'version': '1.0',
               'policies': {'changepoint_sigma': '1.3', 'fps_source': 'median'}}
        model = {'package': 'moseq2-model', 'version': '1.0',
                 'policies': {'sampler_seeding': 'explicit-seed'}}
        records = {'scores.h5': pca, 'model.p': model}
        with self.assertRaises(ProvenanceError) as ctx:
            enforce_consistent_provenance(list(records), records=records)
        message = str(ctx.exception)
        self.assertIn('cross-package', message)
        self.assertIn('no-op', message)
        self.assertIn('moseq2-pca', message)
        self.assertIn('moseq2-model', message)

    def test_two_files_of_the_same_package_still_conflict(self):
        records = {
            'a.h5': {'package': 'moseq2-pca', 'policies': {'changepoint_sigma': '1.3'}},
            'b.h5': {'package': 'moseq2-pca', 'policies': {'changepoint_sigma': '2.0'}},
        }
        with self.assertRaises(ProvenanceError) as ctx:
            enforce_consistent_provenance(list(records), records=records)
        message = str(ctx.exception)
        self.assertIn('changepoint_sigma', message)
        self.assertIn('a.h5', message)
        self.assertIn('b.h5', message)

    def test_provenance_from_model_reads_the_pickled_record(self):
        record = {'package': 'moseq2-model', 'policies': {'sampler_seeding': 'explicit-seed'}}
        self.assertEqual(provenance_from_model({'pipeline_provenance': record}), record)

    def test_provenance_from_unstamped_model_is_none(self):
        self.assertIsNone(provenance_from_model({'labels': [], 'keys': []}))
        self.assertIsNone(provenance_from_model(None))


class TestProvenanceReading(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reads_extraction_path_as_fallback(self):
        path = join(self.tmp, 'extract.h5')
        record = _record(focal_length='pinhole')
        _write_h5(path, record, h5_path='metadata/extraction/pipeline')
        self.assertEqual(read_pipeline_provenance(path), record)

    def test_missing_record_reads_as_none(self):
        path = join(self.tmp, 'plain.h5')
        _write_h5(path, None)
        self.assertIsNone(read_pipeline_provenance(path))

    def test_describe_unstamped_as_unknown(self):
        description = describe_provenance(None)
        self.assertIn('UNKNOWN provenance', description)
        self.assertIn('cannot be inferred', description)

    def test_describe_lists_policies(self):
        text = describe_provenance(_record(focal_length='pinhole'))
        self.assertIn('moseq2-extract', text)
        self.assertIn('focal_length=pinhole', text)


class TestPoolingBoundaryIsGated(TestCase):
    """
    The gate is only useful if it sits on the path real analyses take. These
    check that `scalars_to_dataframe` -- the point where per-session
    extractions become one dataset -- actually refuses inconsistent input.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _index(self, specs):
        """Build a minimal sorted index over synthetic extraction h5s."""
        files = {}
        for i, (group, provenance) in enumerate(specs):
            uuid = f'uuid-{i}'
            path = join(self.tmp, f'session{i}.h5')
            _write_h5(path, provenance)
            files[uuid] = {
                'path': (path, join(self.tmp, f'session{i}.yaml')),
                'uuid': uuid,
                'group': group,
                'metadata': {'SessionName': f'session{i}', 'SubjectName': f'mouse{i}'},
            }
        return {'files': files, 'pca_path': join(self.tmp, 'pca_scores.h5')}

    def test_conflicting_policies_refuse_to_pool(self):
        from moseq2_viz.scalars.util import scalars_to_dataframe

        index = self._index([
            ('WT', _record(focal_length='pinhole')),
            ('5xFAD', _record(focal_length='small-angle')),
        ])
        with self.assertRaises(ProvenanceError) as ctx:
            scalars_to_dataframe(index, disable_output=True)
        self.assertIn("'focal_length'", str(ctx.exception))
        self.assertIn('the scalar dataframe', str(ctx.exception))

    def test_stamped_unstamped_mix_refuses_to_pool(self):
        from moseq2_viz.scalars.util import scalars_to_dataframe

        index = self._index([
            ('WT', _record(focal_length='pinhole')),
            ('5xFAD', None),
        ])
        with self.assertRaises(ProvenanceError):
            scalars_to_dataframe(index, disable_output=True)

    def test_override_gets_past_the_gate(self):
        from moseq2_viz.scalars.util import scalars_to_dataframe

        index = self._index([
            ('WT', _record(focal_length='pinhole')),
            ('5xFAD', _record(focal_length='small-angle')),
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                scalars_to_dataframe(index, disable_output=True,
                                     allow_mixed_provenance=True)
            except ProvenanceError:
                self.fail('allow_mixed_provenance=True must not raise')
            except Exception:
                # the synthetic h5s are not complete extractions, so loading
                # them may fail later. What matters is that it got past the gate.
                pass
        self.assertTrue(
            any('should not be trusted' in str(w.message) for w in caught),
            'override must still report the mismatch loudly')
