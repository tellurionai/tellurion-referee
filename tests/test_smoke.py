"""Hello-world test so CI is green before real code lands."""


def test_packages_import():
    import referee  # noqa: F401
    import tellurion_sim  # noqa: F401
