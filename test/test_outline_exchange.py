import copy
import numpy as np
import pytest
from utils.outline_exchange import lift_outline, world_to_view, compare_outlines, validate_outline_document


def document(kind='candidate', shift=0, heaps=1):
    angle = .3
    transform = np.array([[1,0,0,2], [0,np.cos(angle),-np.sin(angle),-3],
                          [0,np.sin(angle),np.cos(angle),4], [0,0,0,1]])
    scene = dict(dataset_name='fixture', backend='DA3', frame_count=3,
                 world_to_view=transform.ravel().tolist(), display_bounds=[-10,-10,-10,10,10,10])
    regions = []
    for index in range(heaps):
        x = shift + 3 * index
        xz = np.array([[x,0], [x+1,0], [x+1,1], [x,1]])
        world = lift_outline(xz, scene)
        assert np.allclose(world_to_view(world, scene)[:, [0,2]], xz)
        regions.append(dict(id=str(index), vertices_world=world.tolist(), area_m2=1.0))
    return dict(format='viewer-horizontal-outlines', version=1, coordinate_unit='m', kind=kind,
                scene=scene, regions=regions, total_area_m2=float(heaps), complete=True, value_m2=None,
                status='candidate' if kind=='candidate' else 'reference',
                metric='da3_manual_equivalent_stack_outline_candidate' if kind=='candidate' else 'manual_horizontal_outline')


def test_identical_multiple_heaps_and_tilt_roundtrip():
    result = compare_outlines(document(heaps=2), document('manual', heaps=2))
    assert result['union_iou'] == pytest.approx(1)
    assert result['candidate_region_count'] == 2
    assert result['reference_to_candidate_boundary_p95_m'] < 1e-10
    assert result['boundary_sampling_spacing_m'] == .01


def test_translated_equal_area():
    result = compare_outlines(document(shift=.5), document('manual'))
    assert result['relative_area_difference'] == pytest.approx(0)
    assert result['union_iou'] == pytest.approx(1/3)
    assert result['candidate_to_reference_boundary_p95_m'] == pytest.approx(.5)


def test_wrong_scene_and_overlap():
    candidate = document()
    reference = document('manual')
    reference['scene']['dataset_name'] = 'other'
    with pytest.raises(ValueError, match='scene mismatch'):
        compare_outlines(candidate, reference)
    other = copy.deepcopy(candidate['regions'][0]); other['id'] = 'other'
    candidate['regions'].append(other); candidate['total_area_m2'] = 2
    with pytest.raises(ValueError, match='overlap'):
        validate_outline_document(candidate)


@pytest.mark.parametrize('mutate', [
    lambda d: d['regions'][0]['vertices_world'].append(d['regions'][0]['vertices_world'][0]),
    lambda d: d['regions'][0].update(area_m2=5),
    lambda d: d.update(value_m2=1),
    lambda d: d['scene']['world_to_view'].__setitem__(0, 2),
])
def test_invalid_documents(mutate):
    data = document(); mutate(data)
    with pytest.raises(ValueError):
        validate_outline_document(data)


def test_empty_manual_and_empty_comparison():
    reference = document('manual', heaps=0)
    assert validate_outline_document(reference, 'manual') == ()
    with pytest.raises(ValueError, match='reference has no valid regions'):
        compare_outlines(document(), reference)


@pytest.mark.parametrize('declared_area', [0, -1e-12])
def test_tiny_region_requires_positive_declared_area(declared_area):
    data = document('manual')
    vertices = np.array([[0, 0], [.001, 0], [.001, .001], [0, .001]])
    data['regions'][0]['vertices_world'] = lift_outline(vertices, data['scene']).tolist()
    data['regions'][0]['area_m2'] = declared_area
    data['total_area_m2'] = declared_area
    with pytest.raises(ValueError, match='region area_m2'):
        validate_outline_document(data, 'manual')
    data['regions'][0]['area_m2'] = data['total_area_m2'] = 1e-6
    assert len(validate_outline_document(data, 'manual')) == 1


def test_identical_adjacent_regions_use_union_boundary():
    candidate, reference = document(heaps=2), document('manual', heaps=2)
    for data in (candidate, reference):
        view = world_to_view(data['regions'][1]['vertices_world'], data['scene'])
        view[:,0] -= 2
        data['regions'][1]['vertices_world'] = lift_outline(view[:,[0,2]], data['scene']).tolist()
    result = compare_outlines(candidate, reference)
    assert result['union_iou'] == pytest.approx(1)
    assert result['candidate_to_reference_boundary_p95_m'] < 1e-10
    assert result['reference_to_candidate_boundary_p95_m'] < 1e-10


@pytest.mark.parametrize('data', [[], None, dict(document(), scene=[]), dict(document(), regions=[3])])
def test_non_object_input_is_domain_error(data):
    with pytest.raises(ValueError, match='object'):
        validate_outline_document(data)


@pytest.mark.parametrize('field', ['world_to_view', 'display_bounds', 'vertices_world'])
@pytest.mark.parametrize('bad_type', ['object', 'null', 'ragged', 'bool', 'string', 'nonfinite'])
def test_numeric_array_input_boundary(field, bad_type):
    data = document()
    container = data['regions'][0] if field == 'vertices_world' else data['scene']
    original = container[field]
    if bad_type == 'object':
        bad = {}
    elif bad_type == 'null':
        bad = None
    elif bad_type == 'ragged':
        bad = [[1, 2], [3]]
    else:
        bad = copy.deepcopy(original)
        replacement = {'bool': True, 'string': '1.0', 'nonfinite': float('inf')}[bad_type]
        if field == 'vertices_world':
            bad[0][0] = replacement
        else:
            bad[0] = replacement
    container[field] = bad
    with pytest.raises(ValueError, match=field):
        validate_outline_document(data)


@pytest.mark.parametrize('version', [True, '1', 1.0, None, {}])
def test_version_requires_json_integer(version):
    data = document(); data['version'] = version
    with pytest.raises(ValueError, match='version'):
        validate_outline_document(data)


def test_finite_integer_and_float_json_arrays_remain_valid():
    data = document('manual')
    data['scene']['world_to_view'] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    data['regions'][0]['vertices_world'] = [[0,0,0], [1.0,0,0], [1,0,1.0], [0,0,1]]
    assert len(validate_outline_document(data, 'manual')) == 1
