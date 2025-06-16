// server.js
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
app.use(cors());
app.use(bodyParser.json());

let abnormalUsers = [];
const sseClients = [];

// SSE 구독 엔드포인트
app.get('/mcs/alert/stream', (req, res) => {
  res.writeHead(200, {
    Connection: 'keep-alive',
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache'
  });
  res.write('\n');
  sseClients.push(res);

  req.on('close', () => {
    const idx = sseClients.indexOf(res);
    if (idx !== -1) sseClients.splice(idx, 1);
  });
});

function broadcast(payload) {
  const msg = `data: ${JSON.stringify(payload)}\n\n`;
  sseClients.forEach(client => client.write(msg));
}

// 1) React 프론트엔드(더미 사용자 클릭) → 중계 서버 : 새로운 알림 수신
app.post('/mcs/alert/send', (req, res) => {
  const { id, name } = req.body;
  abnormalUsers.push({ id, name });
  console.log(`[Server] New abnormal user added:`, { id, name });

  broadcast({
    type: 'new_alert',
    user: { id, name }
  });

  res.status(200).send({ message: 'Alert sent to clients' });
});

// 2) 중계 서버 자체 조회용
app.get('/mcs/alert/fetch', (req, res) => {
  res.status(200).send({ users: abnormalUsers });
  abnormalUsers = [];
});

// 3) 사용자가 브라우저에서 URL 접속 - OK 응답
app.get('/mcs/:userId/ok', (req, res) => {
  const userId = req.params.userId;
  
  // 사용자 상태 업데이트 및 브로드캐스트
  abnormalUsers = abnormalUsers.filter(u => u.id !== userId);
  console.log(`[Server] User marked OK:`, userId);

  broadcast({
    type: 'status',
    userId,
    status: 'ok'
  });
  
  // 사용자에게 응답 페이지 보여주기
  res.send(`
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
          .container { max-width: 500px; margin: 0 auto; padding: 20px; }
          .success { color: green; }
          h1 { color: #333; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>응답이 완료되었습니다</h1>
          <p class="success">✓ 안전 상태로 업데이트되었습니다.</p>
          <p>이 창은 닫으셔도 됩니다.</p>
        </div>
      </body>
    </html>
  `);
});

// 4) 사용자가 브라우저에서 URL 접속 - EMERGENCY 응답
app.get('/mcs/:userId/emergency', (req, res) => {
  const userId = req.params.userId;
  
  // 사용자 상태 업데이트 및 브로드캐스트
  abnormalUsers = abnormalUsers.filter(u => u.id !== userId);
  console.log(`[Server] User marked EMERGENCY:`, userId);

  broadcast({
    type: 'status',
    userId,
    status: 'emergency'
  });
  
  // 사용자에게 응답 페이지 보여주기
  res.send(`
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
          .container { max-width: 500px; margin: 0 auto; padding: 20px; }
          .emergency { color: red; }
          h1 { color: #333; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>응답이 완료되었습니다</h1>
          <p class="emergency">⚠ 위험 상태로 업데이트되었습니다.</p>
          <p>담당자가 곧 연락드릴 예정입니다. 이 창은 닫으셔도 됩니다.</p>
        </div>
      </body>
    </html>
  `);
});

app.listen(3003, () => {
  console.log('Express server running on http://localhost:3003');
});