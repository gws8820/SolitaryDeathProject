import React, { useState, useEffect } from 'react';
import './App.css';
import { PieChart, Pie, Cell, Label } from 'recharts';
import { useNavigate } from 'react-router-dom';

const COLORS = ['#005bff', '#5c7cfa', '#fab005', '#fa5252'];

function App() {
  const [searchTerm, setSearchTerm] = useState('');
  const [tableData, setTableData] = useState([]);
  const [filter, setFilter] = useState('전체');
  const navigate = useNavigate();

  useEffect(() => {
    fetch('/abnormal_detection.json')
      .then(res => res.json())
      .then(data => {
        const latestMap = new Map();

        data.forEach(item => {
          const key = item.User;
          const timestamp = new Date(`${item.Date}T${item.Type === 'day' ? '12:00:00' : '23:59:59'}`);
          if (!latestMap.has(key) || timestamp > latestMap.get(key).timestamp) {
            latestMap.set(key, { ...item, timestamp });
          }
        });

        const mapped = Array.from(latestMap.values()).map((item, index) => {
          const score = item.Consensus_score;
          let status = '';
          if (score >= 80) status = '심각(80점 이상)';
          else if (score >= 65) status = '주의(65~80점)';
          else if (score >= 50) status = '관심(50~65점)';
          else status = '정상(0~50점)';

          return {
            no: index + 1,
            user: item.User,
            date: item.Date,
            type: item.Type,
            score,
            status,
          };
        });

        setTableData(mapped);
      });
  }, []);

  const filteredTable = tableData.filter(row => filter === '전체' || row.status === filter);

  const statusCounts = {
    '정상(0~50점)': 0,
    '관심(50~65점)': 0,
    '주의(65~80점)': 0,
    '심각(80점 이상)': 0,
  };
  tableData.forEach(row => statusCounts[row.status]++);

  const STATUS_DATA = [
    { name: '정상(0~50점)', value: statusCounts['정상(0~50점)'] },
    { name: '관심(50~65점)', value: statusCounts['관심(50~65점)'] },
    { name: '주의(65~80점)', value: statusCounts['주의(65~80점)'] },
    { name: '심각(80점 이상)', value: statusCounts['심각(80점 이상)'] },
  ];
  const total = STATUS_DATA.reduce((sum, item) => sum + item.value, 0);

  const filteredUsers = tableData
    .filter(user => filter === '전체' || user.status === filter)
    .filter(user => user.user.toLowerCase().includes(searchTerm.toLowerCase()));

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <h1>안부똑똑서비스</h1>
        <ul>
          <li>실시간 상황</li>
        </ul>
      </aside>
      <main className="main-content">
        <div className="dashboard-grid">
          <section className="status-summary">
            <h3>LED 이상탐지 현황</h3>
            <div className="donut-chart-box">
              <div className="donut-chart">
                <PieChart width={260} height={260}>
                  <Pie
                    data={STATUS_DATA}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={110}
                    dataKey="value"
                    startAngle={90}
                    endAngle={-270}
                    labelLine={false}
                    label={({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
                      const RADIAN = Math.PI / 180;
                      const radius = (innerRadius + outerRadius) / 2;
                      const x = cx + radius * Math.cos(-midAngle * RADIAN);
                      const y = cy + radius * Math.sin(-midAngle * RADIAN);
                      return (
                        <text x={x} y={y} fill="#fff" fontSize="13" fontWeight="bold" textAnchor="middle" dominantBaseline="central">
                          {`${(percent * 100).toFixed(0)}%`}
                        </text>
                      );
                    }}
                  >
                    {STATUS_DATA.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index]} />
                    ))}
                    <Label
                      position="center"
                      content={() => (
                        <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" fill="#fff" fontSize="18" fontWeight="700">
                          전체
                          <tspan x="50%" dy="22">{total}명</tspan>
                        </text>
                      )}
                    />
                  </Pie>
                </PieChart>
              </div>
              <ul className="status-legend">
                {STATUS_DATA.map((entry, index) => (
                  <li key={entry.name}>
                    <span className="legend-dot" style={{ backgroundColor: COLORS[index] }}></span>
                    {entry.name} <strong>{entry.value}명</strong>
                  </li>
                ))}
              </ul>
            </div>

            <div style={{ marginBottom: '1px' }}>
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="select-filter"
              >
                <option value="전체">전체</option>
                {STATUS_DATA.map(s => (
                  <option key={s.name} value={s.name}>{s.name}</option>
                ))}
              </select>
            </div>

            <div className="table-scroll">
              <table className="status-table">
                <thead>
                  <tr>
                    <th>번호</th>
                    <th>사용자</th>
                    <th>날짜</th>
                    <th>타입</th>
                    <th>점수</th>
                    <th>위험등급</th>
                    <th>상세</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTable.map((row) => (
                    <tr key={row.no}>
                      <td>{row.no}</td>
                      <td>{row.user}</td>
                      <td>{row.date}</td>
                      <td>{row.type}</td>
                      <td>{row.score.toFixed(1)}</td>
                      <td>{row.status}</td>
                      <td>
                        <button
                          className="detail-btn"
                          onClick={() => navigate(`/user/${row.user}`, { state: row })}
                          title="상세보기"
                        >
                          &gt;
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="user-card-section">
            <h3>유저별 관리</h3>
            <input
              className="user-search-input"
              type="text"
              placeholder="유저번호를 입력하세요."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
            <div className="user-cards-scroll">
              {filteredUsers.map((user, idx) => {
                const statusClass = user.status.includes('심각') ? 'danger'
                                  : user.status.includes('주의') ? 'warning'
                                  : user.status.includes('관심') ? 'caution'
                                  : 'safe';

                const label = user.status.includes('심각') ? '위험'
                              : user.status.includes('주의') ? '주의'
                              : user.status.includes('관심') ? '관심'
                              : '안전';

                return (
                  <div key={user.user} className="user-card">
                    <div className="user-card-info">
                      <span>{`${(idx + 1).toString().padStart(2, '0')}. ${user.user}`}</span>
                      <span className={`status-badge ${statusClass}`}>{label}</span>
                    </div>
                    <button
                      className="alert-btn"
                      onClick={() => navigate(`/user/${user.user}`, { state: user })}
                    >
                      상세 보기
                    </button>
                  </div>
                );
              })}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
