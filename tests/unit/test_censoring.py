from water_analysis.preprocessing.censoring import parse_censored_value


def test_parse_exact_value() -> None:
    result = parse_censored_value("1,25", detection_limit_lower="0,1", detection_limit_upper="")

    assert result.numeric_approx == 1.25
    assert result.censoring_type == "exact"
    assert result.is_censored is False
    assert result.detection_limit_lower == 0.1


def test_parse_left_censored_value() -> None:
    result = parse_censored_value("<0.05")

    assert result.numeric_approx == 0.025
    assert result.censoring_type == "left_censored"
    assert result.censoring_upper_bound == 0.05
    assert result.is_censored is True


def test_parse_right_censored_value() -> None:
    result = parse_censored_value(">1.0")

    assert result.numeric_approx == 1.0
    assert result.censoring_type == "right_censored"
    assert result.censoring_lower_bound == 1.0
    assert result.is_censored is True


def test_parse_interval_value() -> None:
    result = parse_censored_value("0.10-0.30")

    assert result.numeric_approx == 0.2
    assert result.censoring_type == "interval"
    assert result.censoring_lower_bound == 0.1
    assert result.censoring_upper_bound == 0.3


def test_parse_missing_value() -> None:
    result = parse_censored_value("")

    assert result.numeric_approx is None
    assert result.parse_status == "missing"
    assert result.censoring_type == "missing"


def test_parse_scientific_notation_negative_exponent() -> None:
    result = parse_censored_value("5e-05")

    assert result.numeric_approx == 5e-05
    assert result.parse_status == "parsed"
    assert result.censoring_type == "exact"
    assert result.is_censored is False


def test_parse_scientific_notation_positive_exponent() -> None:
    result = parse_censored_value("1.2E+03")

    assert result.numeric_approx == 1200.0
    assert result.censoring_type == "exact"


def test_parse_word_left_qualifier() -> None:
    result = parse_censored_value("менее 0,2")

    assert result.numeric_approx == 0.1
    assert result.censoring_type == "left_censored"
    assert result.censoring_upper_bound == 0.2
    assert result.is_censored is True


def test_parse_word_right_qualifier_case_insensitive() -> None:
    result = parse_censored_value("Больше 50")

    assert result.numeric_approx == 50.0
    assert result.censoring_type == "right_censored"
    assert result.censoring_lower_bound == 50.0
    assert result.is_censored is True
