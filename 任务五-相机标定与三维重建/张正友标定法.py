import numpy as np
import cv2
import glob

#参数初始化与标定板设置
def calibrate_with_extrinsics(chessboard_size=(8, 6), square_size=25, image_dir='calib_photos/*.jpg'):
    #准备对象点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)  #初始化零矩阵,创建N×3的零矩阵，N=棋盘格内角点总数（行×列）,每行存储一个角点的(X,Y,Z)坐标
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size  #生成二维网格索引,例如8×6棋盘格生成0-7和0-5的整数序列,将网格坐标转换为N×2的形式,将索引值转换为物理尺寸

    #存储对象点和图像点
    objpoints = []
    imgpoints = []
    images = glob.glob(image_dir)

    #角点检测
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #检测棋盘格角点，返回检测状态和角点坐标
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            #亚像素级角点精确化，提高定位精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  #最大迭代次数30，误差阈值0.001
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  #(11,11)为搜索窗口尺寸，(-1,-1)为死区半径(自动计算)

            objpoints.append(objp)
            imgpoints.append(corners2)

    #相机标定
    (ret,
     mtx,  #3×3相机内参矩阵
     dist, #畸变系数向量(k1,k2,p1,p2[,k3[,k4,k5,k6]])
     rvecs, #每张图像的旋转向量
     tvecs  #每张图像的平移向量
     ) = cv2.calibrateCamera(
        objpoints,   #3D世界坐标点集
        imgpoints,   #对应的2D图像坐标点集
        gray.shape[::-1],  #图像尺寸(宽×高)
        None,
        None
    )

    #将旋转向量转换为旋转矩阵
    rotation_mats = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]

    #计算重投影误差
    mean_error = 0
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)  #使用标定参数将3D点投影回2D图像
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)  #计算原始点与重投影点的欧氏距离误差
        errors.append(error)
        mean_error += error
    mean_error /= len(objpoints)

    #返回完整标定结果
    return {
        'camera_matrix': mtx,   #3×3相机内参矩阵
        'dist_coeffs': dist,    #畸变系数向量(k1,k2,p1,p2[,k3[,k4,k5,k6]])
        'rotation_vectors': rvecs,  #每张图像的旋转向量
        'rotation_matrices': rotation_mats,  #旋转矩阵
        'translation_vectors': tvecs,   #每张图像的平移向量
        'reprojection_error': mean_error,  #平均重投影误差
        'per_view_errors': errors,
        'image_count': len(objpoints)
    }


if __name__ == "__main__":
    #执行标定
    calibration_data = calibrate_with_extrinsics()

    #打印内参
    print("=== 相机内参 ===")
    print("内参矩阵:\n", calibration_data['camera_matrix'])
    print("畸变系数:\n", calibration_data['dist_coeffs'].ravel())
    print("平均重投影误差:", calibration_data['reprojection_error'], "像素")

    #打印外参(每张图像)
    print("\n=== 相机外参 ===")
    for i in range(calibration_data['image_count']):
        print(f"\n图像 {i + 1} 的外参:")
        print("旋转矩阵:\n", calibration_data['rotation_matrices'][i])
        print("平移向量(mm):\n", calibration_data['translation_vectors'][i].ravel())
