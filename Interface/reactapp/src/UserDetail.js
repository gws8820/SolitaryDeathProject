import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './App.css';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,ReferenceLine
} from 'recharts';

function UserDetail() {
  const location = useLocation();
  const navigate = useNavigate();
  const userId = location.pathname.split('/').pop();

  const [recentStatus, setRecentStatus] = useState('');
  const [scoreTable, setScoreTable] = useState({ headers: [], values: [] });
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    fetch('/abnormal_detection.json')
      .then(res => res.json())
      .then(data => {
        const userData = data.filter(d => d.User === userId);
        if (userData.length === 0) return;

        const latest = userData.sort((a, b) =>
          new Date(`${b.Date}T${b.Type === 'day' ? '12:00:00' : '23:59:59'}`) -
          new Date(`${a.Date}T${a.Type === 'day' ? '12:00:00' : '23:59:59'}`)
        )[0];

        const score = latest.Consensus_score;
        let status = '';
        if (score >= 80) status = '심각(80점 이상)';
        else if (score >= 65) status = '주의(65~80점)';
        else if (score >= 50) status = '관심(50~65점)';
        else status = '정상(0~50점)';
        setRecentStatus(status);

        const dateMap = {};
        userData.forEach(entry => {
          if (!dateMap[entry.Date]) dateMap[entry.Date] = { day: null, night: null };
          dateMap[entry.Date][entry.Type] = entry.Consensus_score;
        });

        const recentDates = Object.keys(dateMap)
          .sort((a, b) => new Date(b) - new Date(a))
          .slice(0, 3);

        const headers = [];
        const values = [];
        recentDates.forEach(date => {
          const day = dateMap[date].day;
          const night = dateMap[date].night;
          headers.push(`${date} 오전`);
          values.push(day != null ? `${day.toFixed(1)}점` : '-');
          headers.push(`${date} 오후`);
          values.push(night != null ? `${night.toFixed(1)}점` : '-');
        });

        headers.push('최근');
        values.push(`${score.toFixed(1)}점`);
        headers.push('업데이트 시간');
        values.push(`${latest.Date} ${latest.Type === 'day' ? '오전' : '오후'}`);

        setScoreTable({ headers, values });

        const chart = userData
          .sort((a, b) => new Date(`${a.Date}T${a.Type === 'day' ? '06:00:00' : '18:00:00'}`) - new Date(`${b.Date}T${b.Type === 'day' ? '06:00:00' : '18:00:00'}`))
          .slice(0, 12)
          .map(entry => ({
            time: `${entry.Date} ${entry.Type === 'day' ? '오전' : '오후'}`,
            percent: entry.Consensus_score
          }));

        setChartData(chart);
      });
  }, [userId]);

  const handleSendAlert = async () => {
    try {
      await fetch(`http://172.30.1.28:3003/mcs/alert/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: userId })
      });
      alert('알림 전송 완료');
    } catch (error) {
      alert('알림 전송 실패');
    }
  };

  const recentStatusClass = (status) => {
    if (status.includes('정상')) return 'normal';
    if (status.includes('관심')) return 'caution';
    if (status.includes('주의')) return 'warning';
    if (status.includes('심각')) return 'danger';
    return '';
  };

  const getStatusLabel = (score) => {
    if (score >= 80) return { text: '심각', className: 'badge danger' };
    if (score >= 65) return { text: '주의', className: 'badge warning' };
    if (score >= 50) return { text: '관심', className: 'badge caution' };
    return { text: '정상', className: 'badge normal' };
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

          <div className="top-card-row">
            <div className="top-card">
              <div className="card-title">🆔 ID</div>
              <div className="card-content">{userId}</div>
            </div>
            <div className={`top-card status-card ${recentStatusClass(recentStatus)}`}>
              <div className="card-title">🩺 최근 상태</div>
              <div className="card-content">{recentStatus}</div>
            </div>
            <div className="top-card button-card">
              <div className="card-title">&nbsp;</div>
              <button className="alert-btn" onClick={handleSendAlert}>🔔 알림 전송</button>
            </div>
          </div>

          <div className="user-records-table">
            <h2>시간대별 감지 기록</h2>
            <table>
              <thead>
                <tr>
                  {scoreTable.headers.map((h, idx) => (
                    <th key={idx}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  {scoreTable.values.map((v, idx) => (
                    <td key={idx}>{v}</td>
                  ))}
                </tr>
                <tr>
                  {scoreTable.values.map((v, idx) => {
                    const numeric = parseFloat(v);
                    const isScore = v.includes('점') && !isNaN(numeric);
                    if (!isScore) return <td key={idx}></td>;

                    const { text, className } = getStatusLabel(numeric);
                    return (
                      <td key={idx}>
                        <span className={className}>{text}</span>
                      </td>
                    );
                  })}
                </tr>
              </tbody>
            </table>
          </div>

          <h2>최근 6일 간의 고독사 위험 점수</h2>
          <div className="user-chart">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={chartData}
                margin={{ top: 10, right: 20, bottom: 50, left: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis
                  dataKey="time"
                  stroke="#aaa"
                  angle={-45}
                  textAnchor="end"
                  tickFormatter={(label) => {
                    const [date, time] = label.split(' ');
                    const [_, month, day] = date.split('-');
                    return `${month}-${day} ${time}`;
                  }}
                />
                <YAxis domain={[0, 100]} tickFormatter={(t) => `${t}점`} stroke="#aaa" />

                {/*수평선 */}
                <ReferenceLine y={50} stroke="red" strokeDasharray="4 2" label={{ value: '위험 임계값 50점', position: 'insideTopLeft', fill: 'red', fontSize: 12 }} />

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
