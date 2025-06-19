// UserDetail.js
import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './App.css';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

const detailData = [
  { time: '6일 전', percent: 26.3 },
  { time: '5일 전', percent: 21.5 },
  { time: '4일 전', percent: 27.6 },
  { time: '3일 전', percent: 22.1},
  { time: '2일 전', percent: 22.4 },
  { time: '1일 전 오전', percent: 78.9},
  { time: '1일 전 오후', percent: 77.3 },
];

const tableData = {
  headers: ['3일 전 오전', '3일 전 오후', '2일 전 오전', '2일 전 오후', '1일 전 오전', '1일 전 오후', '최근', '업데이트 시간'],
  values: ['26.3점', '21.5점', '27.6점', '22.1점', '22.4점', '78.9점', '77.3점', '2025-06-17 00:00']
};

function UserDetail() {
  const location = useLocation();
  const navigate = useNavigate();
  const { id, name, status } = location.state || {};

  const handleSendAlert = async () => {
    try {
      await fetch(`http://172.30.1.28:3003/mcs/alert/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, name })
      });
      alert('알림 전송 완료');
    } catch (error) {
      alert('알림 전송 실패');
    }
  };

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <h1>안부똑똑서비스</h1>
        <ul>
          <li onClick={() => navigate('/')}>실시간 상황</li>
        </ul>
      </aside>

      <main className="detail-content">
        <div className="user-detail-container">
         <section className="user-info-card">
            <p style={{
              fontSize: '25px',
              fontWeight: '700',
              marginBottom: '16px',
              color: '#fff',

              borderRadius: '12px',
              display: 'inline-block'
            }}>
              상세 정보
            </p>

            <div className="user-info-row">
              <div><strong>ID:</strong> {id}</div>
              <div><strong>상태:</strong> {status}</div>
            </div>
            <button className="alert-btn" onClick={handleSendAlert}>알림 전송</button>
          </section>

          <div className="user-records-table">
            <h2>시간대별 감지 기록</h2>
            <table>
              <thead>
                <tr>
                  {tableData.headers.map((h, idx) => (
                    <th key={idx}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  {tableData.values.map((v, idx) => (
                    <td key={idx}>{v}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        <h2>최근 6일 간의 고독사 위험 점수</h2>
          <div className="user-chart">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={detailData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis dataKey="time" stroke="#aaa" />
                <YAxis domain={[0, 100]} tickFormatter={(t) => `${t}%`} stroke="#aaa" />
                <Tooltip
                  formatter={(value) => [`${value}점`, '고독사 위험 점수']}
                  labelFormatter={(label) => `${label}`}
                />
                <Line type="monotone" dataKey="percent" stroke="#0cf" dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

        </div>
      </main>
    </div>
  );
}

export default UserDetail;
