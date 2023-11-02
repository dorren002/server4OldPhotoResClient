from flask import Flask, request, Response
import os, time

from models.detection import DetectionModel
from models.inference import RestorationModel
  
app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = 'tmp'  # 上传文件保存的目录，
app.config['IN_FOLDER'] = app.config['UPLOAD_FOLDER'] + '/input'
app.config['MASK_FOLDER'] = app.config['UPLOAD_FOLDER'] + '/mask'
app.config['OUT_FOLDER'] = app.config['UPLOAD_FOLDER'] + '/output'

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 设置请求体最大允许 16MB
  
@app.route('/detect', methods=['POST'])  
def segAPI():  
    data = request.data # 获取二进制流  

    # 指定保存的文件名  
    timestamp = str(int(time.time()))
    fname = timestamp + '.jpg'
    filename = os.path.join(app.config['IN_FOLDER'], fname)

    # 以二进制模式打开文件并写入数据  
    with open(filename, 'wb') as f: 
        f.write(data)  
    
    # 调用模型接口
    dmHelper = DetectionModel()
    dmHelper.inference(fname)
    
    # 返回处理结果
    return Response(open(os.path.join(app.config['MASK_FOLDER'], fname), 'rb').read(), mimetype='image/jpg')  


@app.route('/restore', methods=['POST'])  
def restoreAPI():
    data = request.data
    
    # 指定保存的文件名  
    timestamp = str(int(time.time()))
    fname = timestamp + '.jpg'
    filename = os.path.join(app.config['IN_FOLDER'], fname)

    # 以二进制模式打开文件并写入数据  
    with open(filename, 'wb') as f: 
        f.write(data)  
    
    # 调用模型接口
    rmHelper = RestorationModel()
    rmHelper.inference(fname)
    
    # 返回处理结果
    return Response(open(os.path.join(app.config['OUT_FOLDER'], fname), 'rb').read(), mimetype='image/jpg')  

if __name__ == '__main__':  
    app.run(host='0.0.0.0', port=2012, debug=True)  # 启动 Flask 应用，并开启调试模式