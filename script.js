const minPartConfidence = 0.2; // 最小の信頼度のしきい値
const color1 = 'aqua'; // video1 のキーポイントとスケルトンの色
const color2 = 'red'; // video2 のキーポイントとスケルトンの色
const lineWidth = 3; // スケルトンの線の太さ
const maxAllowError = 50; // 許容される最大のエラー

// video1 の要素と canvas1 の要素を取得
const video1 = document.getElementById('video1');
const canvas1 = document.getElementById('canvas1');
const contentWidth1 = canvas1.width;
const contentHeight1 = canvas1.height;
const ctx1 = canvas1.getContext('2d');

// video2 の要素と canvas2 の要素を取得
const video2 = document.getElementById('video2');
const canvas2 = document.getElementById('canvas2');
const contentWidth2 = canvas2.width;
const contentHeight2 = canvas2.height;
const ctx2 = canvas2.getContext('2d');

let correct_pose; // 正しいポーズ
let user_pose; // ユーザーのポーズ
let error; // エラー
let intervalId; // setInterval の ID
let score = 0; // スコア

navigator.mediaDevices.getUserMedia({ video: true }) // getUserMedia を使ってカメラのストリームを取得
  .then(stream => {
    video2.srcObject = stream;
    video2.onloadedmetadata = () => {
      resizeCanvas(video2, canvas2);
      startVideo();
    };
  })
  .catch(err => console.error(err));

function resizeCanvas(video, canvas) {
    canvas.width = video.videoWidth; // キャンバスの幅を動画の幅に合わせる
    canvas.height = video.videoHeight; // キャンバスの高さを動画の高さに合わせる
  }
  
document.getElementById('start-button').onclick = function () { // スタートボタンがクリックされたときの処理
  target_score = document.getElementById('score');
  target_score.innerHTML = 'SCORE: ' + String(score);
  target = document.getElementById('good');
  target.innerHTML = '　';
  video1.play(); // video1 を再生
};
  
document.getElementById('stop-button').onclick = function () { // ストップボタンがクリックされたときの処理
  stopLoop();
};

function startVideo() { // 動画の処理を開始する関数
  video1.addEventListener('play', () => { // video1 が再生されたときのイベントリスナー
    window.addEventListener('resize', () => {
      resizeCanvasToWindowSize(canvas1);
      resizeCanvasToWindowSize(canvas2);
    });
    
    intervalId = setInterval(() => {
      posenet.load().then(net => { // posenet.load() でモデルをロード
        return net.estimateMultiplePoses(video1, {
          flipHorizontal: false,
          maxDetections: 2,
          scoreThreshold: 0.6,
          nmsRadius: 20,
        });
      }).then(poses => { // ポーズを推定
        ctx1.clearRect(0, 0, contentWidth1, contentHeight1);
        poses.forEach(({ keypoints }) => {
          drawKeypoints(keypoints, minPartConfidence, ctx1, color1);
          drawSkeleton(keypoints, minPartConfidence, ctx1, color1);
        });

        correct_pose = poses[0]; // 正しいポーズは配列の先頭にあると仮定
        if (user_pose) {
          error = calcAngleError(correct_pose, user_pose); // エラーを計算
          target = document.getElementById('good');

          if (error <= maxAllowError) {
            target.innerHTML = 'GOOD!';
            score = score + 1;
            target_score = document.getElementById('score');
            target_score.innerHTML = 'SCORE: ' + String(score);
          } else {
            target.innerHTML = '　';
          }
        }
      });
    }, 500);
  });

  posenet.load().then(net => { // posenet.load() でモデルをロード
    setInterval(() => {
      net.estimateSinglePose(video2, {
        flipHorizontal: true,
      }).then(pose => { // カメラからのポーズ推定
        ctx2.clearRect(0, 0, contentWidth2, contentHeight2);
        drawKeypoints(pose.keypoints, minPartConfidence, ctx2, color2);
        drawSkeleton(pose.keypoints, minPartConfidence, ctx2, color2);
        user_pose = pose; // ユーザーのポーズを設定
      });
    }, 500);
  });
  
function stopLoop() { // ループを停止する関数
  clearInterval(intervalId);
  video1.pause();
}
  
function toTuple({ y, x }) {
  return [y, x];
}

function drawKeypoints(keypoints, minConfidence, ctx, color, scale = 1) { // キーポイントを描画する関数
  const excludedParts = ['leftEye', 'rightEye', 'leftEar', 'rightEar'];

  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];
    if (keypoint.score < minConfidence || excludedParts.includes(keypoint.part)) {
      continue;
    }
    const { y, x } = keypoint.position;
    drawPoint(ctx, y * scale, x * scale, 3, color);
  }
}

function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawSkeleton(keypoints, minConfidence, ctx, color, scale = 1) {
  const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence);

  adjacentKeyPoints.forEach(adjacentKeypoints => {
    drawSegment(
      toTuple(adjacentKeypoints[0].position),
      toTuple(adjacentKeypoints[1].position),
      color,
      scale,
      ctx
    );
  });
}

function calcAngleError(correct_pose, user_pose) {
  let error = 0;

  error += calcKeypointAngleError(correct_pose, user_pose, 5, 7);
  error += calcKeypointAngleError(correct_pose, user_pose, 6, 8);
  error += calcKeypointAngleError(correct_pose, user_pose, 7, 9);
  error += calcKeypointAngleError(correct_pose, user_pose, 8, 10);
  error += calcKeypointAngleError(correct_pose, user_pose, 11, 13);
  error += calcKeypointAngleError(correct_pose, user_pose, 12, 14);
  error += calcKeypointAngleError(correct_pose, user_pose, 13, 15);
  error += calcKeypointAngleError(correct_pose, user_pose, 14, 16);

  error /= 8;

  return error;
}

function calcKeypointAngleError(correct_pose, user_pose, num1, num2) {
  let error = Math.abs(
    calcKeypointsAngle(correct_pose.keypoints, num1, num2) -
    calcKeypointsAngle(user_pose.keypoints, num1, num2)
  );
  if (error <= 180) {
    return error;
  } else {
    return 360 - error;
  }
}

function calcKeypointsAngle(keypoints, num1, num2) {
  return calcPositionAngle(
    keypoints[num1].position,
    keypoints[num2].position
  );
}

function calcPositionAngle(position1, position2) {
  return calcAngleDegrees(position1.x, position1.y, position2.x, position2.y);
}

function calcAngleDegrees(x1, y1, x2, y2) {
  return Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
}

function resizeCanvasToVideoSize(video, canvas) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function resizeCanvasToWindowSize(canvas) {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function resizeCanvas(video, canvas) {
  if (window.innerWidth > video.videoWidth) {
    resizeCanvasToVideoSize(video, canvas);
  } else {
    resizeCanvasToWindowSize(canvas);
  }
}
}