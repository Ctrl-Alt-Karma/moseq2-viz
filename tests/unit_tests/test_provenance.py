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

from moseq2_viz.util import (EXTRACTION_POOLING_POLICIES,
                             SCALAR_POOLING_POLICIES, ProvenanceError,
                             describe_provenance,
                             enforce_consistent_provenance,
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


def _write_h5(path, provenance=None,
              h5_path='metadata/extraction/pipeline'):
    """Create a minimal h5 with (or without) a provenance record."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('scalars/centroid_x_px', data=[1.0, 2.0, 3.0])
        if provenance is not None:
            f.create_dataset(h5_path, data=json.dumps(provenance))
    return path


def _complete_flip_record(classifier='a' * 64, **updates):
    record = {
        'package': 'moseq2-app',
        'version': '0.1.0',
        'git_sha': 'app-sha',
        'state': 'complete',
        'classifier_sha256': classifier,
        'chunk_size': 4000,
        'chunk_overlap': 0,
        'smoothing': 51,
        'frame_path': 'frames',
        '_journal_sequence': 1,
    }
    record.update(updates)
    return record


def _write_flip_record(path, record, slot='metadata/processing/flip_classifier'):
    with h5py.File(path, 'a') as f:
        f.create_dataset(slot, data=json.dumps(record))


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

    def test_required_extraction_keys_match_documented_snapshot(self):
        # Exact cross-repository comparison lives in pipeline-control, where
        # both locked repositories are required inputs.
        self.assertEqual(set(EXTRACTION_POOLING_POLICIES), set(FULL_POLICIES))

    def test_identical_full_policies_pass_silently(self):
        a, b = self._pair(FULL_POLICIES, FULL_POLICIES)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertTrue(enforce_consistent_provenance(
                [a, b], required_policies=SCALAR_POOLING_POLICIES))
        self.assertEqual([str(w.message) for w in caught], [])

    def test_every_required_policy_is_actually_compared(self):
        """Flip one policy at a time; each must be caught on its own."""
        for key in EXTRACTION_POOLING_POLICIES:
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

class TestProvenanceReading(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reads_extracts_preferred_path(self):
        path = join(self.tmp, 'extract.h5')
        record = _record(focal_length='pinhole')
        _write_h5(path, record, h5_path='metadata/extraction/pipeline')
        observed = read_pipeline_provenance(path)
        self.assertEqual(observed['package'], record['package'])
        self.assertEqual(observed['policies']['focal_length'], 'pinhole')
        self.assertEqual(observed['policies']['flip_correction'], 'not_applied')

    def test_valid_historical_path_is_supported(self):
        path = join(self.tmp, 'historical.h5')
        _write_h5(path, _record(focal_length='pinhole'),
                  h5_path='metadata/pipeline')
        self.assertEqual(
            read_pipeline_provenance(path)['policies']['focal_length'],
            'pinhole')

    def test_semantically_identical_dual_paths_are_accepted(self):
        path = join(self.tmp, 'same.h5')
        record = _record(focal_length='pinhole')
        _write_h5(path, record)
        with h5py.File(path, 'a') as f:
            f.create_dataset('metadata/pipeline', data=json.dumps(record))
        self.assertEqual(
            read_pipeline_provenance(path)['policies']['focal_length'],
            'pinhole')

    def test_conflicting_dual_paths_are_corrupt(self):
        path = join(self.tmp, 'conflict.h5')
        _write_h5(path, _record(focal_length='pinhole'))
        with h5py.File(path, 'a') as f:
            f.create_dataset(
                'metadata/pipeline',
                data=json.dumps(_record(focal_length='small-angle')))
        with self.assertRaisesRegex(ProvenanceError, 'conflicting records'):
            read_pipeline_provenance(path)

    def test_truncated_json_is_corrupt(self):
        path = join(self.tmp, 'truncated.h5')
        with h5py.File(path, 'w') as f:
            f.create_dataset('metadata/extraction/pipeline',
                             data='{"policies": ')
        with self.assertRaisesRegex(ProvenanceError, 'CORRUPT'):
            read_pipeline_provenance(path)

    def test_every_corrupt_input_hard_fails_pooling(self):
        paths = []
        for name in ('a.h5', 'b.h5'):
            path = join(self.tmp, name)
            with h5py.File(path, 'w') as f:
                f.create_dataset('metadata/extraction/pipeline',
                                 data='{"policies": ')
            paths.append(path)
        with self.assertRaisesRegex(ProvenanceError, 'CORRUPT'):
            enforce_consistent_provenance(paths)

    def test_one_corrupt_and_one_valid_hard_fails_pooling(self):
        corrupt = join(self.tmp, 'corrupt.h5')
        with h5py.File(corrupt, 'w') as f:
            f.create_dataset('metadata/extraction/pipeline', data='null')
        valid = _write_h5(
            join(self.tmp, 'valid.h5'), _record(focal_length='pinhole'))
        with self.assertRaisesRegex(ProvenanceError, 'object/mapping'):
            enforce_consistent_provenance([corrupt, valid])

    def test_non_object_json_is_corrupt(self):
        for i, raw in enumerate(('null', '7', '"text"', '[1, 2]')):
            path = join(self.tmp, 'non-object-{}.h5'.format(i))
            with h5py.File(path, 'w') as f:
                f.create_dataset('metadata/extraction/pipeline', data=raw)
            with self.assertRaisesRegex(ProvenanceError, 'object/mapping'):
                read_pipeline_provenance(path)

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


class TestFlipCorrectionPolicy(TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, name, provenance=True):
        record = _record(**FULL_POLICIES) if provenance else None
        return _write_h5(join(self.tmp, name), record)

    def test_complete_and_not_applied_refuse_to_pool(self):
        corrected = self._path('corrected.h5')
        plain = self._path('plain.h5')
        _write_flip_record(corrected, _complete_flip_record())
        with self.assertRaisesRegex(ProvenanceError, 'flip-correction'):
            enforce_consistent_provenance(
                [corrected, plain],
                required_policies=SCALAR_POOLING_POLICIES)

    def test_compatible_complete_records_can_pool(self):
        paths = [self._path('a.h5'), self._path('b.h5')]
        for path in paths:
            _write_flip_record(path, _complete_flip_record())
        self.assertTrue(enforce_consistent_provenance(
            paths, required_policies=SCALAR_POOLING_POLICIES))

    def test_different_complete_records_refuse_to_pool(self):
        a = self._path('a.h5')
        b = self._path('b.h5')
        _write_flip_record(a, _complete_flip_record(classifier='a' * 64))
        _write_flip_record(b, _complete_flip_record(classifier='b' * 64))
        with self.assertRaisesRegex(ProvenanceError, 'flip-correction'):
            enforce_consistent_provenance([a, b])

    def test_incomplete_and_failed_records_hard_fail(self):
        for state in ('in_progress', 'failed'):
            path = self._path(state + '.h5')
            _write_flip_record(path, _complete_flip_record(state=state))
            with self.assertRaisesRegex(ProvenanceError, 'not poolable'):
                enforce_consistent_provenance([path])

    def test_corrupt_flip_record_hard_fails(self):
        path = self._path('corrupt.h5')
        with h5py.File(path, 'a') as f:
            f.create_dataset(
                'metadata/processing/flip_classifier',
                data='{"state": ')
        with self.assertRaisesRegex(ProvenanceError, 'CORRUPT'):
            enforce_consistent_provenance([path])

    def test_non_mapping_flip_record_hard_fails(self):
        for i, raw in enumerate(('null', '7', '"text"', '[1, 2]')):
            path = self._path('non-object-{}.h5'.format(i))
            with h5py.File(path, 'a') as f:
                f.create_dataset(
                    'metadata/processing/flip_classifier', data=raw)
            with self.assertRaisesRegex(ProvenanceError, 'object/mapping'):
                enforce_consistent_provenance([path])

    def test_legacy_latch_only_hard_fails(self):
        path = self._path('legacy.h5')
        with h5py.File(path, 'a') as f:
            f.attrs['flip_classifier_applied'] = True
        with self.assertRaisesRegex(ProvenanceError, 'legacy'):
            enforce_consistent_provenance([path])

    def test_update_sentinel_hard_fails(self):
        path = self._path('sentinel.h5')
        _write_flip_record(path, _complete_flip_record())
        with h5py.File(path, 'a') as f:
            f.attrs['flip_classifier_record_update_in_progress'] = True
        with self.assertRaisesRegex(ProvenanceError, 'sentinel'):
            enforce_consistent_provenance([path])

    def test_unstamped_corrected_and_plain_still_refuse(self):
        corrected = self._path('corrected.h5', provenance=False)
        plain = self._path('plain.h5', provenance=False)
        _write_flip_record(corrected, _complete_flip_record())
        with self.assertRaisesRegex(ProvenanceError, 'flip-correction'):
            enforce_consistent_provenance([corrected, plain])


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
