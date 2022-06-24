import sys
import os
import glob
import json
import shutil
import datetime
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

import streamlit as st
import st_rerun
import SessionState


@st.cache
def read_csv_as_df(path, header=0, names=None, usecols=None):
    """
    :param path: Path to the csv file
    :param header: if it exists, it is 0, otherwise it is None
    :param names: if header does not exists, names are the list of the columns' names
    :param usecols: list of numbers to select specific columns, start with 0
    """
    return pd.read_csv(path, header=header, names=names, usecols=usecols)


# data should be a dictionary and has same keys if append=True
def write_csv(path, data, append=False, **kwargs):
    for field in kwargs:
        data[field] = str(kwargs[field])
    df = pd.DataFrame([data])
    if append:
        df.to_csv(path, mode='a', encoding='utf-8', header=False, index=False)
    else:
        df.to_csv(path, mode='w', encoding='utf-8', index=False)


def load_nifti(nifti_path, pixel_type):
    return sitk.ReadImage(nifti_path, pixel_type)


def label_overlay(img, seg):
    img_uint8 = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
    return sitk.LabelOverlay(img_uint8, seg, opacity=0.25)


def generate_mip_with_labels(nifti_img, nifti_seg, option, num_classes):
    img_array = sitk.GetArrayFromImage(nifti_img)
    seg_array = sitk.GetArrayFromImage(nifti_seg)

    img_mip = []
    seg_mip = []
    mip_issue = False
    for c in range(1, num_classes + 1):
        seg_c = (seg_array == c)
        if option == 'ax':
            sum_seg_c = np.sum(np.sum(seg_c, axis=2), axis=1)
        elif option == 'sag':
            sum_seg_c = np.sum(np.sum(seg_c, axis=1), axis=0)
        else:
            sum_seg_c = np.sum(np.sum(seg_c, axis=2), axis=0)

        if np.any(sum_seg_c):
            nonzero_seg_c = np.argwhere(sum_seg_c > 0)
            start_ind = np.min(nonzero_seg_c)
            end_ind = np.max(nonzero_seg_c)

            if option == 'ax':
                img_mip.append(np.amax(img_array[start_ind:end_ind + 1], axis=0).astype(np.float32))
                seg_mip.append(np.amax(seg_c[start_ind:end_ind + 1], axis=0).astype(np.uint8))
            elif option == 'sag':
                img_mip.append(np.amax(img_array[:, :, start_ind:end_ind + 1], axis=2).astype(np.float32))
                seg_mip.append(np.amax(seg_c[:, :, start_ind:end_ind + 1], axis=2).astype(np.uint8))
            else:
                img_mip.append(np.amax(img_array[:, start_ind:end_ind + 1, :], axis=1).astype(np.float32))
                seg_mip.append(np.amax(seg_c[:, start_ind:end_ind + 1, :], axis=1).astype(np.uint8))

            seg_mip[-1][seg_mip[-1] == 1] = c
            seg_mip[-1][seg_mip[-1] == 0] = 0
        else:
            mip_issue = True
            write_log(logger, '+++ There is a MIP issue (no non-zero prediction)...')
            if option == 'ax':
                img_mip.append(np.zeros((img_array.shape[1], img_array.shape[2])).astype(np.float32))
                seg_mip.append(np.zeros((seg_array.shape[1], seg_array.shape[2])).astype(np.uint8))
            elif option == 'sag':
                img_mip.append(np.zeros((img_array.shape[0], img_array.shape[1])).astype(np.float32))
                seg_mip.append(np.zeros((seg_array.shape[0], seg_array.shape[1])).astype(np.uint8))
            else:
                img_mip.append(np.zeros((img_array.shape[0], img_array.shape[2])).astype(np.float32))
                seg_mip.append(np.zeros((seg_array.shape[0], seg_array.shape[2])).astype(np.uint8))

    nifti_img_mip = [sitk.GetImageFromArray(i_mip) for i_mip in img_mip]
    nifti_seg_mip = [sitk.GetImageFromArray(s_mip) for s_mip in seg_mip]
    for idx in range(len(nifti_img_mip)):
        if option == 'ax':
            nifti_img_mip[idx].CopyInformation(nifti_img[:, :, 0])
            nifti_seg_mip[idx].CopyInformation(nifti_seg[:, :, 0])
        elif option == 'sag':
            nifti_img_mip[idx].CopyInformation(nifti_img[0, :, :])
            nifti_seg_mip[idx].CopyInformation(nifti_seg[0, :, :])
        else:
            nifti_img_mip[idx].CopyInformation(nifti_img[:, 0, :])
            nifti_seg_mip[idx].CopyInformation(nifti_seg[:, 0, :])

    return nifti_img_mip, nifti_seg_mip, mip_issue


def load_subject(state, format_type, format_name):
    subject = {
        'index': state.qc_subject_no,
        'subject_id': state.qc_subject_ids[state.qc_subject_no],
        'qc_csv_folder': os.path.join(state.qc_out_folder, state.qc_subject_ids[state.qc_subject_no]),
        'qc_csv_path': os.path.join(state.qc_out_folder, state.qc_subject_ids[state.qc_subject_no], state.qc_csv_basename)
    }
    os.makedirs(subject['qc_csv_folder'], exist_ok=True)
    img_basename = format_name.split('_')[0]
    img_path = os.path.join(state.qc_img_folder, subject['subject_id'], img_basename + '.nii.gz')
    seg_path = os.path.join(state.qc_seg_folder, subject['subject_id'], state.qc_seg_basename + '.nii.gz')
    if format_type == 'cor' or format_type == 'sag' or format_type == 'ax':
        img = load_nifti(img_path, sitk.sitkFloat32)
        seg = load_nifti(seg_path, sitk.sitkUInt8)
        mip_img, mip_seg, _ = generate_mip_with_labels(img, seg, format_type, len(state.qc_class_names))
        res = [label_overlay(mip_img[idx], mip_seg[idx]) for idx in range(len(mip_img))]
        res = sitk.JoinSeries(res)
    elif format_type == 'ovl':
        img = load_nifti(img_path, sitk.sitkFloat32)
        seg = load_nifti(seg_path, sitk.sitkUInt8)
        res = label_overlay(img, seg)
    elif format_type == 'prd':
        res = load_nifti(seg_path, sitk.sitkUInt8)
    else:
        res = load_nifti(img_path, sitk.sitkFloat32)

    subject[format_name + '_array'] = sitk.GetArrayFromImage(res)
    subject[format_name + '_size'] = res.GetSize()
    subject[format_name + '_spacing'] = res.GetSpacing()
    return subject


@st.cache
def get_subject_ids(subjects_dir):
    paths = glob.glob(os.path.join(subjects_dir, '*', ''))
    subjects = [os.path.basename(os.path.dirname(path)) for path in paths]
    subjects.sort()
    return subjects


def get_next_available(out_folder, subject_ids, qc_csv_basename):
    for idx in range(len(subject_ids)):
        if not os.path.isfile(os.path.join(out_folder, subject_ids[idx], qc_csv_basename)):
            return idx
    return len(subject_ids) - 1


def get_total_checked(out_folder, subject_ids, qc_csv_basename):
    total_checked = {sub: os.path.isfile(os.path.join(out_folder, sub, qc_csv_basename)) for sub in subject_ids}
    total_checked_num = 0
    for key, val in total_checked.items():
        total_checked_num += int(val)
    return total_checked, total_checked_num


def display_image(img_array, size, spacing, format, window=None, level=None, fig_size=(8, 4), axis_on=False):

    dim1 = size[0] * spacing[0]
    dim2 = size[1] * spacing[1]

    window = np.max(img_array) - np.min(img_array) if window is None else window
    level = window / 2 + np.min(img_array) if level is None else level
    low = level - window / 2
    high = level + window / 2

    num_rows = 1
    num_cols = img_array.shape[0]
    fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)
    if format == 'sag':
        for i in range(num_rows * num_cols):
            axs[i].imshow(img_array[i], origin='lower', cmap='gray', clim=(low, high), extent=(0, dim1, 0, dim2))
    elif format == 'ax':
        for i in range(num_rows * num_cols):
            axs[i].imshow(img_array[i], origin='lower', cmap='gray', clim=(low, high), extent=(0, dim1, dim2, 0))
    else:
        for i in range(num_rows * num_cols):
            axs[i].imshow(img_array[i], origin='lower', cmap='gray', clim=(low, high), extent=(0, dim1, 0, dim2))

    if not axis_on:
        for i in range(num_rows * num_cols):
            axs[i].axis('off')

    fig.tight_layout()

    return fig


def display_volume(vol_array, size, spacing, x=None, y=None, z=None, window=None, level=None, fig_size=(8, 4), axis_on=False):

    width = size[0] * spacing[0]
    height = size[1] * spacing[1]
    depth = size[2] * spacing[2]
      
    x = size[0] // 2 if x is None else x
    y = size[1] // 2 if y is None else y
    z = size[2] // 2 if z is None else z
    
    window = np.max(vol_array) - np.min(vol_array) if window is None else window
    level = window / 2 + np.min(vol_array) if level is None else level
    low = level - window / 2
    high = level + window / 2

    fig, axs = plt.subplots(1, 3, figsize=fig_size)

    axs[0].imshow(vol_array[:, y, :], origin='lower', cmap='gray', clim=(low, high), extent=(0, width, 0, depth))
    axs[1].imshow(vol_array[z, :, :], origin='lower', cmap='gray', clim=(low, high), extent=(0, width, height, 0))
    axs[2].imshow(vol_array[:, :, x], origin='lower', cmap='gray', clim=(low, high), extent=(0, height, 0, depth))

    if not axis_on:
        for i in range(len(axs)):
            axs[i].axis('off')

    fig.tight_layout()

    return fig


def correct_subject_no(state):
    if state.qc_subject_no < 0:
        state.qc_subject_no = 0
    elif state.qc_subject_no >= len(state.qc_subject_ids):
        state.qc_subject_no = len(state.qc_subject_ids) - 1
    write_log(logger, 'correct_subject_no: {0}'.format(state.qc_subject_no))


def format_subject_ids(state):
    total_checked, _ = get_total_checked(state.qc_out_folder, state.qc_subject_ids, state.qc_csv_basename)
    return ['{0} [{1} - {2}]'.format(state.qc_subject_ids[idx], idx, bool(total_checked[state.qc_subject_ids[idx]]))
            for idx in range(len(state.qc_subject_ids))]


def settings_check(state):
    state.qc_fig_axis_on = st.sidebar.checkbox('Figure Axis On', value=False)
    format_name = st.sidebar.selectbox('Format:', state.qc_format_names)
    state.qc_format_index = state.qc_format_names.index(format_name)
    write_log(logger, 'qc_fig_axis_on: {0}'.format(state.qc_fig_axis_on))
    write_log(logger, 'qc_format_name: {0}'.format(format_name))
    write_log(logger, 'qc_format_index: {0}'.format(state.qc_format_index))


def prev_button_pressed(state):
    write_log(logger, 'previous_button_pressed')
    state.qc_run_id += 1
    state.qc_subject_no -= 1
    write_log(logger, 'qc_subject_no: {0}'.format(state.qc_subject_no))
    st_rerun.rerun()


def save_qualities(state, cur_subject, button_name):
    state.qc_csv_entry['index'] = cur_subject['index']
    state.qc_csv_entry['subject_id'] = cur_subject['subject_id']
    write_log(logger, 'saving_qualities [{0}]: {1}'.format(button_name, state.qc_csv_entry))
    write_csv(cur_subject['qc_csv_path'], state.qc_csv_entry)


def next_button_pressed(state, cur_subject):
    write_log(logger, 'next_button_pressed')
    save_qualities(state, cur_subject, 'next')
    state.qc_run_id += 1
    state.qc_subject_no += 1
    write_log(logger, 'state.qc_subject_no: {0}'.format(state.qc_subject_no))
    st_rerun.rerun()


def save_button_pressed(state, cur_subject):
    write_log(logger, 'save_button_pressed')
    save_qualities(state, cur_subject, 'save')
    st_rerun.rerun()


def retrieve_qualities_from_csv(state, cur_subject):
    if os.path.isfile(cur_subject['qc_csv_path']):
        state.qc_csv_entry = pd.read_csv(cur_subject['qc_csv_path'], header=0, names=None, usecols=None).to_dict('records')[-1]
    else:
        state.qc_csv_entry = {}

    write_log(logger, 'retrieve_qualities_from_csv [csv_path]: {0}'.format(cur_subject['qc_csv_path']))
    write_log(logger, 'retrieve_qualities_from_csv [qualities]: {0}'.format(state.qc_csv_entry))


def update_flag_it(st_element, state):
    val_flag_it = 'flag_it' in state.qc_csv_entry and state.qc_csv_entry['flag_it']
    state.qc_csv_entry['flag_it'] = st_element.checkbox('Flag It', value=val_flag_it, key=state.qc_run_id)


def update_check_again(st_element, state):
    val_check_again = 'check_again' in state.qc_csv_entry and state.qc_csv_entry['check_again']
    state.qc_csv_entry['check_again'] = st_element.checkbox('Check Again', value=val_check_again, key=state.qc_run_id)


def update_qualities(st_element, state):
    for cname in state.qc_class_names:
        if cname not in state.qc_csv_entry:
            option_index = 0
        else:
            option_index = state.qc_option_names.index(state.qc_csv_entry[cname])
        state.qc_csv_entry[cname] = st_element[cname].radio(cname, state.qc_option_names, index=option_index, key=state.qc_run_id)

    write_log(logger, 'state.qc_csv_entry: {0}'.format(state.qc_csv_entry))


def update_subject_info(st_element, state, cur_subject):
    _, total_checked_num = get_total_checked(state.qc_out_folder, state.qc_subject_ids, state.qc_csv_basename)
    processed_string = '{:.3f}'.format(100 * float(total_checked_num) / len(state.qc_subject_ids))
    subject_info_string = 'Subject ID: {0} [index: {1} - processed: {2}%]'.format(cur_subject['subject_id'],
                                                                                cur_subject['index'],
                                                                                processed_string)
    write_log(logger, 'subject_info: {0}'.format(subject_info_string))
    write_log(logger, 'qc_csv_path: {0}'.format(cur_subject['qc_csv_path']))
    write_log(logger, 'total_checked_num: {0}'.format(total_checked_num))

    st_element.text(subject_info_string)

    return subject_info_string


def update_graph(st_graph, st_sliders, state, cur_subject, format_index, format_type, format_name):
    write_log(logger, 'update_graph')
    vol_array = cur_subject[format_name + '_array']
    size = cur_subject[format_name + '_size']
    spacing = cur_subject[format_name + '_spacing']

    cor = ax = sag = -1
    if format_type == 'org' or format_type == 'prd' or format_type == 'ovl':
        cor = st_sliders[0].slider('Coronal Location:', min_value=0, max_value=size[1], value=size[1]//2, step=1, key=state.qc_run_id)
        ax = st_sliders[1].slider('Axial Location:', min_value=0, max_value=size[2], value=size[2]//2, step=1, key=state.qc_run_id)
        sag = st_sliders[2].slider('Sagittal Location:', min_value=0, max_value=size[0], value=size[0]//2, step=1, key=state.qc_run_id)
        fig = display_volume(vol_array, size, spacing, x=sag, y=cor, z=ax, axis_on=state.qc_fig_axis_on)
    else:
        fig = display_image(vol_array, size, spacing, format_type, axis_on=state.qc_fig_axis_on)

    write_log(logger, 'subject_id: {0}, index: {1}'.format(cur_subject['subject_id'], cur_subject['index']))
    write_log(logger, 'format_index: {0}, format_type: {1} format_name: {2}'.format(format_index, format_type, format_name))
    write_log(logger, 'cor: {0}, ax: {1}, sag: {2}'.format(cor, ax, sag))
    write_log(logger, 'qc_csv_path [Exists: {0}]: {1}'.format(os.path.isfile(cur_subject['qc_csv_path']), cur_subject['qc_csv_path']))

    if os.path.isfile(cur_subject['qc_csv_path']):
        fig.patch.set_facecolor('xkcd:mint green')

    st_graph.pyplot(fig)


def update_subject_id_selectbox(state):
    write_log(logger, 'update_subject_id_selectbox')
    st_extra_space_1 = st.sidebar.text('----------------------------------')
    formatted_subject_ids = format_subject_ids(state)
    formatted_subject_id = st.sidebar.selectbox('Subject ID [Index - Checked]:', formatted_subject_ids, index=state.qc_subject_no)
    state.qc_subject_no = formatted_subject_ids.index(formatted_subject_id)
    write_log(logger, 'qc_subject_no: {0}'.format(state.qc_subject_no))
    write_log(logger, 'formatted_subject_id: {0}'.format(formatted_subject_id))


def main(state):
    correct_subject_no(state)
    settings_check(state)
    update_subject_id_selectbox(state)

    format_index = state.qc_format_index
    format_type = state.qc_format_types[format_index]
    format_name = state.qc_format_names[format_index]
    cur_subject = load_subject(state, format_type, format_name)

    st_subject_info = st.empty()
    st_sliders = st.beta_columns(3)
    st_graph = st.empty()

    st_extra_space_2 = st.sidebar.text('----------------------------------')
    st_cols_flagging = st.sidebar.beta_columns(2)
    st_quality_radio_buttons = {}
    for cname in state.qc_class_names:
        st_quality_radio_buttons[cname] = st.sidebar.empty()
    st_extra_space_3 = st.sidebar.text('----------------------------------')

    cols_buttons = st.sidebar.beta_columns(3)
    with cols_buttons[0]:
        prev_pressed = st.button('<-Prev')
        if prev_pressed:
            prev_button_pressed(state)

    with cols_buttons[1]:
        save_pressed = st.button('Save')
        if save_pressed:
            save_button_pressed(state, cur_subject)

    with cols_buttons[2]:
        next_pressed = st.button('Next->')
        if next_pressed:
            next_button_pressed(state, cur_subject)

    retrieve_qualities_from_csv(state, cur_subject)
    update_flag_it(st_cols_flagging[0], state)
    update_check_again(st_cols_flagging[1], state)

    update_qualities(st_quality_radio_buttons, state)
    update_subject_info(st_subject_info, state, cur_subject)
    update_graph(st_graph, st_sliders, state, cur_subject, format_index, format_type, format_name)


def get_log_file(dir_path=None, basename=None):
    file_name, file_extension  = os.path.splitext(__file__)
    if dir_path is None and basename is None:
        log_file_path = file_name + '_log.txt'
    elif dir_path is not None and basename is None:
        log_file_path = os.path.join(dir_path, os.path.basename(file_name) + '_log.txt')
    elif dir_path is None and basename is not None:
        log_file_path = os.path.join(os.path.dirname(file_name), basename)
    else:
        log_file_path = os.path.join(dir_path, basename)
    return log_file_path


def write_log(logger, message):
    log_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f ') + message
    print(log_string)
    logger.write(log_string + '\n')


def load_qc_config(qc_config_path):
    if os.path.isfile(qc_config_path):
        with open(qc_config_path) as f:
            qc_config = json.load(f)
        return qc_config
    else:
        st.error('No QC config file found: {0}'.format(qc_config_path))
        return None


if __name__ == '__main__':
    qc_config_path = os.path.join(os.path.dirname(__file__), 'segQC_config.json')
    qc_config = load_qc_config(qc_config_path)

    if qc_config is not None:
        qc_img_folder = os.path.abspath(qc_config['img_folder'])
        if 'seg_folder' not in qc_config or qc_config['seg_folder'] is None:
            qc_seg_folder = qc_img_folder
        else:
            qc_seg_folder = os.path.abspath(qc_config['seg_folder'])
        if 'out_folder' not in qc_config or qc_config['out_folder'] is None:
            qc_out_folder = qc_img_folder
        else:
            qc_out_folder = os.path.abspath(qc_config['out_folder'])
        os.makedirs(qc_out_folder, exist_ok=True)

        qc_img_basenames = qc_config['img_basenames']
        qc_seg_basename  = qc_config['seg_basename']
        qc_class_names   = qc_config['class_names']
        qc_option_names  = qc_config['qc_options']
        qc_csv_basename  = qc_config['qc_csv_basename']

        qc_format_types = []
        qc_format_names = []
        for bname in qc_img_basenames:
            qc_format_types.append('org')
            qc_format_names.append(bname)
            qc_format_types.append('ovl')
            qc_format_names.append(bname + '_ovl')
            qc_format_types.append('cor')
            qc_format_names.append(bname + '_mip_cor')
            qc_format_types.append('sag')
            qc_format_names.append(bname + '_mip_sag')
            qc_format_types.append('ax')
            qc_format_names.append(bname + '_mip_ax')
        qc_format_types.append('prd')
        qc_format_names.append(qc_seg_basename)

        shutil.copy2(qc_config_path, qc_out_folder)

        log_file_name = get_log_file(dir_path=qc_out_folder, basename=os.path.basename(qc_out_folder) + '_log.txt')
        if os.path.isfile(log_file_name):
            logger = open(log_file_name, 'a')
        else:
            logger = open(log_file_name, 'w')

        st.set_page_config(layout="wide")
        write_log(logger, '============================ THIS IS A RERUN ============================')
        write_log(logger, 'qc_config_path: {0}'.format(qc_config_path))
        write_log(logger, 'qc_img_folder: {0}'.format(qc_img_folder))
        write_log(logger, 'qc_seg_folder: {0}'.format(qc_seg_folder))
        write_log(logger, 'qc_out_folder: {0}'.format(qc_out_folder))
        write_log(logger, 'qc_img_basenames: {0}'.format(qc_img_basenames))
        write_log(logger, 'qc_seg_basename: {0}'.format(qc_seg_basename))
        write_log(logger, 'qc_class_names: {0}'.format(qc_class_names))
        write_log(logger, 'qc_option_names: {0}'.format(qc_option_names))
        write_log(logger, 'qc_csv_basename: {0}'.format(qc_csv_basename))
        write_log(logger, 'qc_format_types: {0}'.format(qc_format_types))
        write_log(logger, 'qc_format_names: {0}'.format(qc_format_names))

        qc_subject_ids = get_subject_ids(qc_img_folder)
        write_log(logger, 'number_of_subject_ids: {0}'.format(len(qc_subject_ids)))
        qc_last_idx = get_next_available(qc_out_folder, qc_subject_ids, qc_csv_basename)
        write_log(logger, 'qc_last_idx: {0}'.format(qc_last_idx))
        qc_state = SessionState.get(qc_run_id=0,
                                    qc_img_folder=qc_img_folder,
                                    qc_seg_folder=qc_seg_folder,
                                    qc_out_folder=qc_out_folder,
                                    qc_img_basenames=qc_img_basenames,
                                    qc_seg_basename=qc_seg_basename,
                                    qc_class_names=qc_class_names,
                                    qc_option_names=qc_option_names,
                                    qc_csv_basename=qc_csv_basename,
                                    qc_subject_ids=qc_subject_ids,
                                    qc_csv_entry={},
                                    qc_format_types=qc_format_types,
                                    qc_format_names=qc_format_names,
                                    qc_format_index=0,
                                    qc_subject_no=qc_last_idx,
                                    qc_fig_axis_on=False)

        main(qc_state)
        logger.close()
