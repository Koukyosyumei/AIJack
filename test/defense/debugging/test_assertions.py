import pytest


def test_assert_multibox():
    from aijack.defense.debugging.assertions import (
        MultiBoxAssertionError,
        assert_multibox,
    )

    boxes_without_error = [
        [219.17691040039062, 407.9515380859375,
            346.6560363769531, 873.8331298828125],
        [673.4752197265625, 400.6047668457031, 810.0, 875.171875],
    ]
    boxes_with_error = [
        [219.17691040039062, 407.9515380859375,
            346.6560363769531, 873.8331298828125],
        [50.10575866699219, 395.4058532714844,
            239.3400115966797, 913.1495361328125],
        [673.4752197265625, 400.6047668457031, 810.0, 875.171875],
        [14.653461456298828, 219.9162139892578, 810.0, 810.0142822265625],
        [0.084969162940979, 553.3789672851562,
            64.70587921142578, 874.105224609375],
    ]

    assert_multibox(boxes_without_error)

    with pytest.raises(MultiBoxAssertionError) as e:
        assert_multibox(boxes_with_error)
        assert str(e.value) == "Box 0 overlaps more than 2 other boxes."


def test_multibox_exception():
    import cv2

    from aijack.defense.debugging.assertions import FlickerException, except_flicker

    img = cv2.imread("test/demodata/bus.jpg")

    boxes_1 = [
        [219.17691040039062, 407.9515380859375,
            346.6560363769531, 873.8331298828125],
        [50.10575866699219, 395.4058532714844,
            239.3400115966797, 913.1495361328125],
        [673.4752197265625, 400.6047668457031, 810.0, 875.171875],
        [14.653461456298828, 219.9162139892578, 810.0, 810.0142822265625],
        [0.084969162940979, 553.3789672851562,
            64.70587921142578, 874.105224609375],
    ]

    boxes_2 = [
        [50.10575866699219, 395.4058532714844,
            239.3400115966797, 913.1495361328125],
        [673.4752197265625, 400.6047668457031, 810.0, 875.171875],
        [14.653461456298828, 219.9162139892578, 810.0, 810.0142822265625],
        [0.084969162940979, 553.3789672851562,
            64.70587921142578, 874.105224609375],
    ]

    boxes_3 = [
        [219.17691040039062, 407.9515380859375,
            346.6560363769531, 873.8331298828125]
    ]

    boxes_of_frames_with_error = [boxes_1, boxes_2, boxes_3]
    boxes_of_frames_without_error = [boxes_1, boxes_2, boxes_2]
    frames = [img, img, img]

    except_flicker(frames, boxes_of_frames_without_error)

    with pytest.raises(FlickerException) as e:
        except_flicker(frames, boxes_of_frames_with_error)
        assert str(
            e.values) == "Flickering detected between frame -1 and frame -3"
