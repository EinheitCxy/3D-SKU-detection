"""Match summaries must preserve file IDs and every detection observation."""
import torch
from src.deduplicate_detections import parse_all_matches, build_global_mapping
from utils.sku_matching_system import SKUMatchingSystem
from utils.config import SKUMatchingConfig
from test.test_web_viewer_export import _classification


def test_noncontiguous_file_ids_survive_summary_roundtrip(tmp_path, monkeypatch):
    import utils.sku_matching_system as module
    monkeypatch.setattr(module, 'visualize_results', lambda *args: None)
    system = object.__new__(SKUMatchingSystem)
    system.config = SKUMatchingConfig.for_3d_mapping(device='cpu')
    system.config.save_json = False
    for ref, target in [(0, 1), (1, 2)]:
        system.config.output_dir = str(tmp_path / str(ref))
        system._post_process_results(
            {target: [{'object_id': 0, 'target_obj_id': 0, 'correspondence_ratio': .9,
                       'matched_points': 9, 'total_points': 10}]},
            None, torch.zeros((3, 3, 2, 2)), [], ref, [], ['0.jpg', '2.jpg', '4.jpg'])
    edges = {(m['ref_idx'], m['target_idx']) for m in parse_all_matches(tmp_path)}
    assert edges == {(0, 2), (2, 4)}
    from src.improved_sku_analyzer import ImprovedSKUCountAnalyzer
    pairs = ImprovedSKUCountAnalyzer(str(tmp_path), str(tmp_path)).analyze_with_filtering()['pairs']
    assert {(m['ref_idx'], m['target_idx']) for m in pairs} == edges


def test_global_mapping_does_not_drop_unlinked_removed_detection():
    obj = {'position': [0, 0, 10, 10], 'classification': _classification()}
    mapping = build_global_mapping(
        [{'ref_idx': 0, 'ref_id': 0, 'target_idx': 1, 'target_id': 0, 'hit_ratio': .9}],
        {0: {0}, 1: set()}, {0: [obj], 1: [obj, obj]}, [0, 1])
    observations = [(o['image_id'], o['object_id']) for values in mapping.values() for o in values]
    assert sorted(observations) == [(0, 0), (1, 0), (1, 1)]
    assert sum(not o['removed'] for values in mapping.values() for o in values) == 2


def test_summary_rejects_partial_match_group(tmp_path):
    import pytest
    folder = tmp_path / '0'; folder.mkdir()
    (folder / 'matching_summary.txt').write_text(
        'Matched ref 0 → target 0 (hit ratio: 0.90 9/10)\n'
        'Matching objects between reference image 0 and target image 2\n'
        'Found 2 matches in image 2\n')
    with pytest.raises(ValueError, match='expected 2 matches'):
        parse_all_matches(tmp_path)


def test_reverse_last_to_first_survives_empty_first_summary(tmp_path):
    (tmp_path / '0').mkdir(); (tmp_path / '0' / 'matching_summary.txt').write_text('Found 0 matches across 0 images\n')
    folder = tmp_path / '2'; folder.mkdir()
    (folder / 'matching_summary.txt').write_text(
        'Matched ref 0 → target 0 (hit ratio: 0.90 9/10)\n'
        'Matching objects between reference image 4 and target image 0\n'
        'Found 1 matches in image 0\n')
    assert [(m['ref_idx'], m['target_idx']) for m in parse_all_matches(tmp_path)] == [(4, 0)]


def test_dedup_files_agree_with_global_mapping_after_conflicting_edges(tmp_path):
    import json
    from src.deduplicate_detections import deduplicate_sequence, resolve_dataset_paths
    dataset = tmp_path / 'dataset'; detections = dataset / 'detections_results'; detections.mkdir(parents=True)
    obj = {'position': [0, 0, 10, 10], 'classification': _classification()}
    for i, objects in [(0, [obj]), (1, [obj, obj])]:
        (detections / f'{i}.json').write_text(json.dumps({'classes': {}, 'objects': objects}))
    output = tmp_path / 'out'; folder = output / 'dataset/output_3dmapping_da3/0'; folder.mkdir(parents=True)
    (folder / 'matching_summary.txt').write_text(
        'Matched ref 0 → target 0 (hit ratio: 0.90 9/10)\n'
        'Matched ref 0 → target 1 (hit ratio: 0.80 8/10)\n'
        'Matching objects between reference image 0 and target image 1\n'
        'Found 2 matches in image 1\n')
    deduplicate_sequence(resolve_dataset_paths(dataset), output_root=output, algorithm='3d', backend='da3', output_subdir='dedup_detections', same_names=True)
    target = output / 'dataset/dedup_detections'
    mapping = json.loads((target / 'global_mapping.json').read_text())
    assert sum(len(v) for v in mapping.values()) == 3
    assert len(json.loads((target / '1.json').read_text())['objects']) == 1


def test_accuracy_uses_source_reference_id_not_directory_slot(tmp_path):
    from accuracy_annotation import AccuracyAnnotator
    folder = tmp_path / '1'; folder.mkdir()
    summary = folder / 'matching_summary.txt'
    summary.write_text('Reference image file ID: 2\n'
        'Matched ref 0 → target 0 (hit ratio: 0.90 9/10)\n'
        'Matching objects between reference image 2 and target image 4\n'
        'Found 1 matches in image 4\n')
    annotator = AccuracyAnnotator()
    annotator.ground_truth = {'3_to_5': [{'reference_id': 0, 'target_id': 0}]}
    annotator.load_vggt_results(str(summary))
    assert len(annotator.vggt_results['3_to_5']) == 1


def test_mapping_rejects_edge_to_missing_detection():
    import pytest
    obj = {'position': [0, 0, 10, 10], 'classification': _classification()}
    with pytest.raises(ValueError, match='unknown detection'):
        build_global_mapping([{'ref_idx': 0, 'ref_id': 0, 'target_idx': 1, 'target_id': 2}],
            {0: {0}, 1: {0}}, {0: [obj], 1: [obj]}, [0, 1])
