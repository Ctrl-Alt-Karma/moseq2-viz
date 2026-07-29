import joblib
import unittest
import numpy as np
import pandas as pd
from unittest import TestCase
from cytoolz import merge_with
from moseq2_viz.util import parse_index, read_yaml
from moseq2_viz.model.util import parse_model_results, h5_to_dict
from moseq2_viz.scalars.util import star_valmap, convert_pxs_to_mm, is_legacy, \
    generate_empty_feature_dict, convert_legacy_scalars, get_scalar_map, get_scalar_triggered_average, \
    nanzscore, _pca_matches_labels, process_scalars, scalars_to_dataframe, \
    compute_all_pdf_data, compute_mouse_dist_to_center, h5_filepath_from_sorted, compute_syllable_position_heatmaps

class TestScalarUtils(TestCase):

    def test_star_valmap(self):
        model_fit = 'data/mock_model.p'
        index_file = 'data/test_index_crowd.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'

        model_data = parse_model_results(model_fit)
        lbl_dict = {}
        labels = model_data['labels']
        for k, v in zip(model_data['keys'], labels):
            lbl_dict[k] = np.asarray(v)

        scores_idx = h5_to_dict(index_data['pca_path'], 'scores_idx')

        def merger(d):
            return merge_with(tuple, scores_idx, d)

        out = star_valmap(np.isnan, merger(lbl_dict))

        assert isinstance(out, dict)
        for v in out.values():
            assert all(np.isnan(v)) == False

    def test_convert_pxs_to_mm(self):
        # NOTE: this test previously recomputed the production formula in its own
        # body and asserted the two matched, so it would have passed for ANY
        # focal-length formula -- including the small-angle approximation it was
        # actually pinning. Assert against independently derived values instead.
        coords = np.asarray([[0, 0], [256, 212], [512, 424]])
        resolution = (512, 424)
        field_of_view = (70.6, 60)
        true_depth = 673.1

        test_coords = convert_pxs_to_mm(coords, resolution, field_of_view, true_depth)

        # pinhole projection: f = (resolution / 2) / tan(fov / 2)
        expected = np.asarray([[-476, -388], [0, 0], [476, 388]])
        np.testing.assert_array_equal(test_coords, expected)
        assert not np.isnan(test_coords).any()

    def test_convert_pxs_to_mm_matches_camera_intrinsics(self):
        # The implied focal lengths must be consistent with the Kinect2's
        # factory-calibrated depth intrinsics (fx = fy ~= 368 at 512x424).
        # The small-angle approximation gave fx=415.5 / fy=404.9, i.e. 10-13%
        # high, which underestimated every millimetre displacement.
        resolution, field_of_view = (512, 424), (70.6, 60)
        expected_fw = resolution[0] / (2 * np.tan(np.deg2rad(field_of_view[0] / 2)))
        expected_fh = resolution[1] / (2 * np.tan(np.deg2rad(field_of_view[1] / 2)))

        # recover the per-pixel scale from a known one-pixel offset
        true_depth = 673.1
        base = np.asarray([[256.0, 212.0]])
        shifted = np.asarray([[257.0, 213.0]])
        step = (convert_pxs_to_mm(shifted, resolution, field_of_view, true_depth)
                - convert_pxs_to_mm(base, resolution, field_of_view, true_depth))[0]

        np.testing.assert_allclose(step[0], true_depth / expected_fw, rtol=1e-6)
        np.testing.assert_allclose(step[1], true_depth / expected_fh, rtol=1e-6)

        # and those focal lengths sit within 2% of the factory intrinsics
        assert abs(expected_fw - 368.1) / 368.1 < 0.02
        assert abs(expected_fh - 368.1) / 368.1 < 0.02

    def test_is_legacy(self):
        true_test = {'centroid_x': [1], 'area': 2}
        false_test = {'centroid_x_mm': [1], 'area_mm': 2}

        assert is_legacy(true_test) == True
        assert is_legacy(false_test) == False

    def test_generate_empty_feature_dict(self):

        num_frames = 900
        empty_dict = generate_empty_feature_dict(num_frames)

        for v in empty_dict.values():
            np.testing.assert_array_equal(v, np.zeros((abs(num_frames),), dtype='float32'))

    def test_convert_legacy_scalars(self):
        test_file = 'data/proc/results_00.h5'
        out_feat = convert_legacy_scalars(test_file)
        assert isinstance(out_feat, dict)
        assert is_legacy(out_feat) == False
        self.assertRaises(KeyError, convert_legacy_scalars, test_file, force=True)

        # setting up values to convert
        out_feat['height_ave_mm'] = [1]
        out_feat['width'] = [1]
        out_feat['width_mm'] = [1]
        out_feat['width_px'] = [1]
        out_feat['length'] = [1]
        out_feat['length_mm'] = [1]
        out_feat['length_px'] = [1]
        out_feat['area'] = [1]
        out_feat['area_mm'] = [1]
        out_feat['area_px'] = [1]

        feats = convert_legacy_scalars(out_feat, force=True)

        assert feats != out_feat

    def test_get_scalar_map(self):
        index_file = 'data/test_index_crowd.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i in range(len(index_data['files'])):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'

        test_scalar_map = get_scalar_map(index_data)
        scalar_keys = ['angle', 'area_mm', 'area_px', 'centroid_x_mm', 'centroid_x_px',
                       'centroid_y_mm', 'centroid_y_px', 'height_ave_mm', 'length_mm',
                       'length_px', 'velocity_2d_mm', 'velocity_2d_px', 'velocity_3d_mm',
                       'velocity_theta', 'width_mm', 'width_px']

        for v in test_scalar_map.values():
            assert list(v) == scalar_keys
        assert len(test_scalar_map.keys()) == 2
        assert isinstance(test_scalar_map, dict)

        self.assertRaises(KeyError, get_scalar_map, index_data, force_conversion=True)


    def test_scalar_triggered_average(self):
        index_file = 'data/test_index_crowd.yaml'
        model_fit = 'data/mock_model.p'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i in range(len(index_data['files'])):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'

        model_data = parse_model_results(model_fit)
        lbl_dict = {}
        labels = model_data['labels']
        for k, v in zip(model_data['keys'], labels):
            lbl_dict[k] = np.asarray(v)

        max_syllable = 40

        test_scalar_map = get_scalar_map(index_data)
        test_syll_avg = get_scalar_triggered_average(test_scalar_map, lbl_dict,
                                                     max_syllable=max_syllable)

        avg_keys = ['velocity_2d_mm', 'velocity_3d_mm', 'width_mm',
                    'length_mm', 'height_ave_mm', 'angle']

        assert avg_keys == list(test_syll_avg.keys())
        for v in test_syll_avg.values():
            assert len(v) == max_syllable
        assert isinstance(test_syll_avg, dict)

        # primitive zscore test
        test_syll_z_avg = get_scalar_triggered_average(test_scalar_map, lbl_dict,
                                               max_syllable=max_syllable, zscore=True)

        assert test_syll_avg.values() != test_syll_z_avg.values()


    def test_nanzscore(self):
        data = np.array([[1,2,3,4,5],[np.nan,np.nan,np.nan,np.nan,np.nan],[6,7,8,9,10]])
        nanz = nanzscore(data)
        print(nanz)
        assert all(nanz[0] < 0)
        assert np.isnan(nanz[1]).all() == True
        assert all(nanz[2] > 0)

    def test_pca_matches_labels(self):
        pca = [1,2,3,4,5,6,7,8,9,10]
        labels = [0,1,2,3,4,5,6,7,8,9]

        assert _pca_matches_labels(pca,labels) == True
        labels = [0,1,2,3]
        assert _pca_matches_labels(pca,labels) == False

    def test_process_scalars(self):
        index_file = 'data/test_index_crowd.yaml'
        scalar_keys = ['angle', 'area_mm', 'area_px', 'centroid_x_mm', 'centroid_x_px',
                       'centroid_y_mm', 'centroid_y_px', 'height_ave_mm', 'length_mm',
                       'length_px', 'velocity_2d_mm', 'velocity_2d_px', 'velocity_3d_mm',
                       'velocity_theta', 'width_mm', 'width_px']

        num_frames = 908
        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i in range(len(index_data['files'])):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'

        test_scalar_map = get_scalar_map(index_data)
        mapped = process_scalars(test_scalar_map, scalar_keys)

        assert isinstance(mapped, dict)
        assert len(mapped.keys()) == 2

        z_mapped = process_scalars(test_scalar_map, scalar_keys, zscore=True)
        assert isinstance(z_mapped, dict)

        for v1, v2 in zip(mapped.values(), z_mapped.values()):
            assert v1.all() != v2.all()
            assert np.asarray(v1).shape == (16, num_frames)
            assert np.asarray(v2).shape == (16, num_frames)

    def test_scalars_to_dataframe(self):
        index_file = 'data/test_index.yaml'

        df_cols = ['angle', 'area_mm', 'area_px', 'centroid_x_mm', 'centroid_x_px',
                   'centroid_y_mm', 'centroid_y_px', 'height_ave_mm', 'length_mm',
                   'length_px', 'velocity_2d_mm', 'velocity_2d_px', 'velocity_3d_mm',
                   'velocity_theta', 'width_mm', 'width_px',
                   'dist_to_center_px', 'group', 'uuid', 'h5_path', 'timestamps',
                   # the trailing metadata columns come out in the order they
                   # appear in the session yaml, which is alphabetical here
                   'frame index', 'SessionName', 'StartTime', 'SubjectName']

        total_frames = 1800

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i in range(len(index_data['files'])):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'
            index_data['files'][i]['path'][1] = 'data/proc/results_00.yaml'

        scalar_df = scalars_to_dataframe(index_data)

        assert isinstance(scalar_df, pd.DataFrame)
        assert all(scalar_df.columns == df_cols)
        assert scalar_df.shape == (total_frames, len(df_cols))

    def test_compute_all_pdf_data(self):

        index_file = 'data/test_index.yaml'

        _, sorted_index = parse_index(index_file)
        scalar_df = scalars_to_dataframe(sorted_index)

        test_pdfs, groups, sessions, subjectNames = compute_all_pdf_data(scalar_df)

        assert len(test_pdfs) == len(groups) == len(sessions) == len(subjectNames)
        for i in range(len(test_pdfs)):
            assert test_pdfs[i].shape == (20, 20)

    def test_compute_syllable_position_heatmaps(self):

        test_index = 'data/test_index.yaml'
        model_path = 'data/test_model.p'

        _, sorted_index = parse_index(test_index)
        # deleting file key to "bad extraction example"
        del sorted_index['files']['ae8a9d45-7ad9-4048-963f-ca4931125fcd']

        scalar_df = scalars_to_dataframe(sorted_index, model_path=model_path)

        default_test_hists = compute_syllable_position_heatmaps(scalar_df,
                                                                syllable_key='labels (usage sort)',
                                                                syllables=range(10),
                                                                centroid_keys=['centroid_x_mm', 'centroid_y_mm'],
                                                                normalize=True,
                                                                bins=20)

        assert isinstance(default_test_hists, pd.Series)

        # 10 found syllable heatmaps per grouping
        assert len(default_test_hists) == 10

    def test_compute_mouse_dist_to_center(self):

        test_index = 'data/test_index.yaml'

        _, sorted_index = parse_index(test_index)

        files = sorted_index['files']

        uuids = list(files.keys())
        dset = h5_to_dict(h5_filepath_from_sorted(files[uuids[0]]), path='scalars')

        # Get ROI shape to compute distance to center
        roi = h5_to_dict(h5_filepath_from_sorted(files[uuids[0]]), path='metadata/extraction/roi')['roi']

        centroid_x_px = dset['centroid_x_px']
        centroid_y_px = dset['centroid_y_px']

        dist_to_center = compute_mouse_dist_to_center(roi, centroid_x_px, centroid_y_px)

        assert len(dist_to_center) == 900

if __name__ == '__main__':
    unittest.main()

class TestDeprecatedScalars(TestCase):
    """velocity_3d_px mixed pixel and millimetre units; it must not be usable."""

    def test_not_produced_in_new_feature_dict(self):
        from moseq2_viz.scalars.util import generate_empty_feature_dict
        feats = generate_empty_feature_dict(10)
        assert 'velocity_3d_px' not in feats
        # the dimensionally valid velocities survive untouched
        for keep in ('velocity_2d_px', 'velocity_2d_mm', 'velocity_3d_mm'):
            assert keep in feats

    def test_explicit_request_raises_actionable_error(self):
        from moseq2_viz.scalars.util import check_scalar_is_usable
        with self.assertRaises(ValueError) as ctx:
            check_scalar_is_usable('velocity_3d_px')
        msg = str(ctx.exception)
        # the message must name the replacements, not just refuse
        assert 'velocity_3d_mm' in msg and 'velocity_2d_px' in msg

    def test_valid_scalars_are_not_blocked(self):
        from moseq2_viz.scalars.util import check_scalar_is_usable
        for ok in ('velocity_2d_px', 'velocity_2d_mm', 'velocity_3d_mm', 'area_mm'):
            check_scalar_is_usable(ok)  # must not raise

    def test_legacy_numeric_column_is_dropped_with_warning(self):
        import warnings as _w
        from moseq2_viz.scalars.util import drop_deprecated_scalars
        df = pd.DataFrame({'velocity_3d_px': [1.0, 2.0], 'velocity_3d_mm': [3.0, 4.0]})
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            out = drop_deprecated_scalars(df)
        assert 'velocity_3d_px' not in out.columns
        assert 'velocity_3d_mm' in out.columns
        assert len(caught) >= 1

    def test_legacy_conversion_does_not_recreate_it(self):
        # convert_legacy_scalars(force=True) must not resurrect the mixed-unit key
        from moseq2_viz.scalars.util import convert_legacy_scalars
        n = 20
        old = {
            'centroid_x': np.linspace(100, 120, n),
            'centroid_y': np.linspace(100, 110, n),
            'width': np.full(n, 30.0), 'length': np.full(n, 60.0),
            'area': np.full(n, 1500.0), 'height_ave': np.full(n, 40.0),
            'angle': np.zeros(n),
        }
        feats = convert_legacy_scalars(old, force=True)
        assert 'velocity_3d_px' not in feats
        assert 'velocity_3d_mm' in feats and np.isfinite(feats['velocity_3d_mm']).all()

    def test_downstream_analysis_works_without_the_key(self):
        """
        Analysis helpers must not assume velocity_3d_px is present. This walks
        a scalar dataframe that never had it through the summarising paths.
        """
        from moseq2_viz.scalars.util import (compute_all_pdf_data,
                                             compute_mean_syll_scalar)

        n = 60
        df = pd.DataFrame({
            'centroid_x_mm': np.linspace(-40, 40, n),
            'centroid_y_mm': np.linspace(-30, 30, n),
            'velocity_2d_px': np.abs(np.sin(np.arange(n))),
            'velocity_2d_mm': np.abs(np.sin(np.arange(n))) * 1.86,
            'velocity_3d_mm': np.abs(np.cos(np.arange(n))) * 1.86,
            'height_ave_mm': np.full(n, 40.0),
            'area_mm': np.full(n, 1500.0),
            'group': ['WT'] * (n // 2) + ['5xFAD'] * (n // 2),
            'uuid': ['u0'] * (n // 2) + ['u1'] * (n // 2),
            'SubjectName': ['m0'] * (n // 2) + ['m1'] * (n // 2),
            'SessionName': ['s0'] * (n // 2) + ['s1'] * (n // 2),
            'labels (usage sort)': np.tile(np.arange(6), n // 6),
            'syllable index': np.arange(n),
            'frame index': np.arange(n),
        })
        assert 'velocity_3d_px' not in df.columns

        pdfs, groups, uuids, subjects = compute_all_pdf_data(df)
        assert len(pdfs) == 2 and len(groups) == 2

        out = compute_mean_syll_scalar(df, scalar='velocity_2d_mm')
        assert 'velocity_2d_mm' in out.columns
        assert np.isfinite(out['velocity_2d_mm'].dropna()).all()


class TestPinholeProjection(TestCase):
    def test_focal_lengths_and_pixel_scale(self):
        from moseq2_viz.scalars.util import convert_pxs_to_mm
        res, fov, depth = (512, 424), (70.6, 60), 673.1
        fx = res[0] / (2 * np.tan(np.deg2rad(fov[0] / 2)))
        fy = res[1] / (2 * np.tan(np.deg2rad(fov[1] / 2)))
        np.testing.assert_allclose(fx, 361.56, atol=0.05)
        np.testing.assert_allclose(fy, 367.19, atol=0.05)

        base = np.asarray([[256.0, 212.0]])
        dx = (convert_pxs_to_mm(np.asarray([[257.0, 212.0]]), res, fov, depth)
              - convert_pxs_to_mm(base, res, fov, depth))[0][0]
        dy = (convert_pxs_to_mm(np.asarray([[256.0, 213.0]]), res, fov, depth)
              - convert_pxs_to_mm(base, res, fov, depth))[0][1]
        np.testing.assert_allclose(dx, 1.862, atol=0.005)
        np.testing.assert_allclose(dy, 1.833, atol=0.005)

        # the small-angle approximation would NOT satisfy these
        bad_fx = res[0] / (2 * np.deg2rad(fov[0] / 2))
        assert abs(bad_fx - 361.56) > 0.05

    def test_image_centre_maps_to_origin(self):
        from moseq2_viz.scalars.util import convert_pxs_to_mm
        out = convert_pxs_to_mm(np.asarray([[256.0, 212.0]]), (512, 424), (70.6, 60), 673.1)
        np.testing.assert_array_equal(out[0], [0.0, 0.0])

    def test_legacy_force_conversion_reaches_the_pinhole_path(self):
        """
        Legacy scalars re-converted through viz must land on the same
        millimetre scale as a fresh extraction, not the small-angle one.
        """
        from moseq2_viz.scalars.util import convert_legacy_scalars

        n = 4
        # legacy layout: unsuffixed centroid/width/length/area/height_ave
        legacy = {
            'centroid_x': np.asarray([256.0, 257.0, 258.0, 259.0]),
            'centroid_y': np.asarray([212.0, 212.0, 212.0, 212.0]),
            'width': np.full(n, 20.0),
            'length': np.full(n, 30.0),
            'area': np.full(n, 600.0),
            'height_ave': np.full(n, 25.0),
            'angle': np.zeros(n),
        }
        out = convert_legacy_scalars(legacy, force=True, true_depth=673.1)

        # image centre is the origin; each subsequent pixel is one pinhole step
        step = 673.1 / (512 / (2 * np.tan(np.deg2rad(70.6 / 2))))
        np.testing.assert_allclose(step, 1.8616, atol=0.0005)
        np.testing.assert_allclose(out['centroid_x_mm'],
                                   np.asarray([0.0, 1.0, 2.0, 3.0]) * step,
                                   atol=1e-6)
        np.testing.assert_array_equal(out['centroid_y_mm'], np.zeros(n))

        # the small-angle approximation gives a visibly smaller step, so this
        # assertion fails if the conversion regresses
        small_angle_step = 673.1 / (512 / (2 * np.deg2rad(70.6 / 2)))
        assert abs(step - small_angle_step) > 0.15

    def test_legacy_force_conversion_matches_extract(self):
        """viz's forced legacy path and extract must agree to floating point."""
        from moseq2_viz.scalars.util import convert_legacy_scalars
        from moseq2_extract.util import convert_pxs_to_mm as extract_conv

        xs = np.asarray([0.0, 100.0, 256.0, 511.0])
        ys = np.asarray([0.0, 90.0, 212.0, 423.0])
        legacy = {
            'centroid_x': xs, 'centroid_y': ys,
            'width': np.full(4, 20.0), 'length': np.full(4, 30.0),
            'area': np.full(4, 600.0), 'height_ave': np.full(4, 25.0),
            'angle': np.zeros(4),
        }
        out = convert_legacy_scalars(legacy, force=True, true_depth=673.1)
        expected = extract_conv(np.stack([xs, ys], axis=1), true_depth=673.1)

        np.testing.assert_allclose(out['centroid_x_mm'], expected[:, 0], rtol=1e-12)
        np.testing.assert_allclose(out['centroid_y_mm'], expected[:, 1], rtol=1e-12)
