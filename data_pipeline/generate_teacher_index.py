import os
import pickle

from teacher_index import build_teacher_course_index, write_teacher_course_index


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    history_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../course-data/history"))

    with open(os.path.join(data_dir, "data_to_index_dict.pkl"), "rb") as data_to_index_file:
        data_to_index_dict = pickle.load(data_to_index_file)

    with open(os.path.join(data_dir, "index_to_data_dict.pkl"), "rb") as course_data_file:
        course_data_dict = pickle.load(course_data_file)

    with open(os.path.join(data_dir, "topic_class_map.pkl"), "rb") as topic_class_map_file:
        topic_class_map = pickle.load(topic_class_map_file)

    teacher_course_index = build_teacher_course_index(
        history_dir=history_dir,
        data_to_index_dict=data_to_index_dict,
        course_data_dict=course_data_dict,
        topic_class_map=topic_class_map,
    )

    output_path = os.path.join(data_dir, "teacher_course_index.pkl")
    write_teacher_course_index(output_path, teacher_course_index)
    print(f"Wrote teacher course index to {output_path}")
    print(f"Indexed {len(teacher_course_index)} teachers")


if __name__ == "__main__":
    main()
