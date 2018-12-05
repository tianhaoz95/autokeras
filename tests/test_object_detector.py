from autokeras.object_detector import ObjectDetector
from tests.common import TEST_TEMP_DIR, clean_dir


def test_object_detector():
    od = ObjectDetector(True)
    od.load()
    results = od.predict('tests/resources/images_test/object_detection.jpg', output_file_path=TEST_TEMP_DIR)
    assert isinstance(results, list)
    clean_dir(TEST_TEMP_DIR)
