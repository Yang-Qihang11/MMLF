def load_pcd(self, pcd_path):
    pts = []
    f = open(pcd_path, 'r')
    data = f.readlines()
    f.close()

    line = data[9].strip('\n')
    pts_num = eval(line.split(' ')[-1])

    for line in data[11:]:
        line = line.strip('\n')
        xyzi = line.split(' ')
        x, y, z, i = [eval(i) for i in xyzi[:4]]
        pts.append([x, y, z, i])

    assert len(pts) == pts_num
    res = np.zeros((pts_num, len(pts[0])), dtype=np.float)
    for i in range(pts_num):
        res[i] = pts[i]

    return res


def scale_to_255(self, a, min, max, dtype=np.uint8):
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def calc_xyz(self, data):
    center_x = (data[16] + data[19] + data[22] + data[25]) / 4.0
    center_y = (data[17] + data[20] + data[23] + data[26]) / 4.0
    center_z = (data[18] + data[21] + data[24] + data[27]) / 4.0
    return center_x, center_y, center_z


def calc_hwl(self, data):
    height = (data[15] - data[27])
    width = math.sqrt(math.pow((data[17] - data[26]), 2) + math.pow((data[16] - data[25]), 2))
    length = math.sqrt(math.pow((data[17] - data[20]), 2) + math.pow((data[16] - data[19]), 2))
    return height, width, length


def calc_yaw(self, data):
    angle = math.atan2(data[17] - data[26], data[16] - data[25])

    if (angle < -1.57):
        return angle + 3.14 * 1.5
    else:
        return angle - 1.57


def cls_type_to_id(self, data):
    type = data[1]
    if type not in model_params['classes']:
        return -1
    return model_params['classes'].index(type)


def calc_angle(self, im, re):
    """
    param: im(float): imaginary parts of the plural
    param: re(float): real parts of the plural
    return: The angle at which the objects rotate
    around the Z axis in the velodyne coordinate system
    """
    if re > 0:
        return np.arctan(im / re)
    elif im < 0:
        return -np.pi + np.arctan(im / re)
    else:
        return np.pi + np.arctan(im / re)


def load_label(self, label_path):
    lines = [line.rstrip() for line in open(label_path)]
    label_list = []

    for line in lines:
        data = line.split(' ')
        data[4:] = [float(t) for t in data[4:]]
        type = data[1]
        if type not in model_params['classes']:
            continue
        label = np.zeros([8], dtype=np.float32)
        label[0], label[1], label[2] = self.calc_xyz(data)
        label[3], label[4], label[5] = self.calc_hwl(data)
        label[6] = self.calc_yaw(data)
        label[7] = self.cls_type_to_id(data)
        label_list.append(label)

    return np.array(label_list)


def transform_bev_label(self, label):
    image_width = (self.y_max - self.y_min) / self.voxel_size
    image_height = (self.x_max - self.x_min) / self.voxel_size

    boxes_list = []
    boxes_num = label.shape[0]

    for i in range(boxes_num):
        center_x = (-label[i][1] / self.voxel_size).astype(np.int32) - int(np.floor(self.y_min / self.voxel_size))
        center_y = (-label[i][0] / self.voxel_size).astype(np.int32) + int(np.ceil(self.x_max / self.voxel_size))
        width = label[i][4] / self.voxel_size
        height = label[i][5] / self.voxel_size

        left = center_x - width / 2
        right = center_x + width / 2
        top = center_y - height / 2
        bottom = center_y + height / 2
        if ((left > image_width) or right < 0 or (top > image_height) or bottom < 0):
            continue
        if (left < 0):
            center_x = (0 + right) / 2
            width = 0 + right
        if (right > image_width):
            center_x = (image_width + left) / 2
            width = image_width - left
        if (top < 0):
            center_y = (0 + bottom) / 2
            height = 0 + bottom
        if (bottom > image_height):
            center_y = (top + image_height) / 2
            height = image_height - top

        box = [center_x, center_y, width, height, label[i][6], label[i][7]]
        boxes_list.append(box)

    while len(boxes_list) < 300:
        boxes_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return np.array(boxes_list, dtype=np.float32)


def transform_bev_image(self, pts):
    x_points = pts[:, 0]
    y_points = pts[:, 1]
    z_points = pts[:, 2]
    i_points = pts[:, 3]

    # convert to pixel position values
    x_img = (-y_points / self.voxel_size).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / self.voxel_size).astype(np.int32)  # y axis is -x in LIDAR

    # shift pixels to (0, 0)
    x_img -= int(np.floor(self.y_min / self.voxel_size))
    y_img += int(np.floor(self.x_max / self.voxel_size))

    # clip height value
    pixel_values = np.clip(a=z_points, a_min=self.z_min, a_max=self.z_max)

    # rescale the height values
    pixel_values = self.scale_to_255(pixel_values, min=self.z_min, max=self.z_max)

    # initalize empty array
    x_max = math.ceil((self.y_max - self.y_min) / self.voxel_size)
    y_max = math.ceil((self.x_max - self.x_min) / self.voxel_size)

    # Height Map & Intensity Map & Density Map
    height_map = np.zeros((y_max, x_max), dtype=np.float32)
    intensity_map = np.zeros((y_max, x_max), dtype=np.float32)
    density_map = np.zeros((y_max, x_max), dtype=np.float32)

    for k in range(0, len(pixel_values)):
        if pixel_values[k] > height_map[y_img[k], x_img[k]]:
            height_map[y_img[k], x_img[k]] = pixel_values[k]
        if i_points[k] > intensity_map[y_img[k], x_img[k]]:
            intensity_map[y_img[k], x_img[k]] = i_points[k]

        density_map[y_img[k], x_img[k]] += 1

    for j in range(0, y_max):
        for i in range(0, x_max):
            if density_map[j, i] > 0:
                density_map[j, i] = np.minimum(1.0, np.log(density_map[j, i] + 1) / np.log(64))

    height_map /= 255.0
    intensity_map /= 255.0

    rgb_map = np.zeros((y_max, x_max, 3), dtype=np.float32)
    rgb_map[:, :, 0] = density_map  # r_map
    rgb_map[:, :, 1] = height_map  # g_map
    rgb_map[:, :, 2] = intensity_map  # b_map

    return rgb_map


def filter_roi(self, pts):
    mask = np.where((pts[:, 0] >= self.x_min) & (pts[:, 0] <= self.x_max) &
                    (pts[:, 1] >= self.y_min) & (pts[:, 1] <= self.y_max) &
                    (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max))
    pts = pts[mask]

    return pts