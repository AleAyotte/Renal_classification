from Patient import Patient
import argparse


def argument_parser():
    """
        A parser the get the name of the experiment that we want to do
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient_id', type=str, default="Penn-028")
    parser.add_argument('--crop_config', type=int, default=0)
    parser.add_argument('--slice', type=int, default=-1)
    parser.add_argument('--orientation', type=str, default='axial')
    return parser.parse_args()


_path = "/home/alex/Data/Corrected/"

if __name__ == "__main__":

    args = argument_parser()
    crop_shape = [69, 65, 15] if args.crop_config == 0 else [89, 87, 19]
    patient_id = "Kidney-" + args.patient_id

    print("# *********************************************** "
          "\n#             Visualize the patient "
          "\n# ***********************************************")
    pat = Patient(patient_id, _path)

    t1 = pat.get_t1()
    t2 = pat.get_t2()
    print("t1 metadata: {}".format(t1.get_metadata()))
    print("t1 roi_measure: {}".format(t1.get_roi_measure()))
    print("t2 metadata: {}".format(t2.get_metadata()))
    print("t2 roi_measure: {}".format(t2.get_roi_measure()))

    pat.plot_image_and_roi(slice_orientation=args.orientation, slice_t1=args.slice, slice_t2=args.slice)

    print("# *********************************************** "
          "\n#                    Apply N4 "
          "\n# ***********************************************")
    pat.apply_n4(save=False)

    t1 = pat.get_t1()
    t2 = pat.get_t2()
    print("t1 metadata: {}".format(t1.get_metadata()))
    print("t1 roi_measure: {}".format(t1.get_roi_measure()))
    print("t2 metadata: {}".format(t2.get_metadata()))
    print("t2 roi_measure: {}".format(t2.get_roi_measure()))

    pat.plot_image_and_roi(slice_orientation=args.orientation, slice_t1=args.slice, slice_t2=args.slice)

    print("# ***********************************************"
          "\n#               Resample and crop"
          "\n# ***********************************************")
    pat.resample_and_crop(resample_params=[1.1, 1.1, 5.0],
                          crop_shape=crop_shape,
                          interp_type=0,
                          register=True,
                          save=False)

    t1 = pat.get_t1()
    t2 = pat.get_t2()
    print("t1 metadata: {}".format(t1.get_metadata()))
    print("t1 roi_measure: {}".format(t1.get_roi_measure()))
    print("t2 metadata: {}".format(t2.get_metadata()))
    print("t2 roi_measure: {}".format(t2.get_roi_measure()))

    pat.plot_image_and_roi(slice_orientation=args.orientation, slice_t1=args.slice, slice_t2=args.slice)

    print("# ***********************************************"
          "\n#                  Apply z norm"
          "\n# ***********************************************")
    pat.apply_znorm()

    t1 = pat.get_t1()
    t2 = pat.get_t2()
    print("t1 metadata: {}".format(t1.get_metadata()))
    print("t1 roi_measure: {}".format(t1.get_roi_measure()))
    print("t2 metadata: {}".format(t2.get_metadata()))
    print("t2 roi_measure: {}".format(t2.get_roi_measure()))

    pat.plot_image_and_roi(slice_orientation=args.orientation, slice_t1=args.slice, slice_t2=args.slice)
