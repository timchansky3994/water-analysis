from water_analysis.domain.point_code import parse_point_code


def test_parse_point_code_preserves_all_parts() -> None:
    point_code = parse_point_code("00000000001.10110.0010")

    assert point_code is not None
    assert point_code.full_point_code == "00000000001.10110.0010"
    assert point_code.oktmo == "00000000001"
    assert point_code.point_type_code == "10110"
    assert point_code.point_number == "0010"


def test_parse_point_code_rejects_invalid_values() -> None:
    assert parse_point_code("") is None
    assert parse_point_code("00000000001.10110") is None
    assert parse_point_code("00000000001..0010") is None
