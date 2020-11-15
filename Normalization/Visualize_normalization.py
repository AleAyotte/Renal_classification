From Patient import Patient
import argparse

def argument_parser():
    """
        A parser the get the name of the experiment that we want to do
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient_id', type=str, default="Penn-028")
    parser.add_argument('crop_config', type=int, default=0)
    return parser.parse_args()

_path = "E:/WORKSPACE_RadiomicsComputation/Kidney/Corrected"

if __name__ == "__main__":

    args = argument_parser()
    crop_shape = [69, 65, 15] if args.crop_config == 0 else [89, 87, 19]
    patient_id = "Kidney-" + args.patient_id

    # ***********************************************
    #             Visualize the patient
    # ***********************************************
    pat = Patient(patient_id, _path, "Penn", "Train")

    t1 = pat.get_t1()
    t2 = pat.get_t2()
    print("t1 metadata: {}".format(t1.get_metadata()))
    print("t1 roi_measure: {}".format(t1.get_roi_measure()))
    print("t2 metadata: {}".format(t2.get_metadata()))
    print("t2 roi_measure: {}".format(t2.get_roi_measure()))

    pat.plot_image_and_roi()

    # ***********************************************
    #                    Apply N4
    # ***********************************************
    pat.apply_n4(save=False)

    t1 = pat.get_t1()
    t2 = pat.get_t2()
    print("t1 metadata: {}".format(t1.get_metadata()))
    print("t1 roi_measure: {}".format(t1.get_roi_measure()))
    print("t2 metadata: {}".format(t2.get_metadata()))
    print("t2 roi_measure: {}".format(t2.get_roi_measure()))

    pat.plot_image_and_roi()

    # ***********************************************
    #               Resample and crop
    # ***********************************************
    pat.resample_and_crop(resample_params=[1.1, 1.1, 5.0],
                          crop_shape=crop_shape,
                          interp_type=0,
                          save=False)

    t1 = pat.get_t1()
    t2 = pat.get_t2()
    print("t1 metadata: {}".format(t1.get_metadata()))
    print("t1 roi_measure: {}".format(t1.get_roi_measure()))
    print("t2 metadata: {}".format(t2.get_metadata()))
    print("t2 roi_measure: {}".format(t2.get_roi_measure()))

    pat.plot_image_and_roi()

    # ***********************************************
    #               Resample and crop
    # ***********************************************
    pat.apply_znorm()

    t1 = pat.get_t1()
    t2 = pat.get_t2()
    print("t1 metadata: {}".format(t1.get_metadata()))
    print("t1 roi_measure: {}".format(t1.get_roi_measure()))
    print("t2 metadata: {}".format(t2.get_metadata()))
    print("t2 roi_measure: {}".format(t2.get_roi_measure()))

    pat.plot_image_and_roi()