// App.js
import React, { useState, useEffect } from 'react';
import './App.css';

const DUMMY_USERS = [
  { id: '121311', name: '서건우' },
  { id: '203131', name: '김혜수' },
  { id: '333210', name: '이지우' },
  { id: '303214', name: '콩코르줄오치르밧' },
  { id: '404512', name: '박민지' },
  { id: '505613', name: '최현우' },
  { id: '606714', name: '정수연' },
  { id: '707815', name: '강태영' },
  { id: '808916', name: '윤서연' },
  { id: '909017', name: '임도현' },
];

function App() {
  const [responses, setResponses] = useState({});
  const [loadingUsers, setLoadingUsers] = useState(new Set());

  // SSE 구독: 중계 서버로부터 새 알림·응답 수신
  useEffect(() => {
    const es = new EventSource('http://172.30.1.28:3003/mcs/alert/stream');

    es.onmessage = e => {
      const data = JSON.parse(e.data);
      console.log('[Frontend ← SSE]', data);

      if (data.type === 'new_alert') {
        setLoadingUsers(prev => new Set([...prev, data.user.id]));
      }
      else if (data.type === 'status') {
        setResponses(prev => ({
          ...prev,
          [data.userId]: data.status
        }));
        setLoadingUsers(prev => {
          const newSet = new Set(prev);
          newSet.delete(data.userId);
          return newSet;
        });
      }
    };

    es.onerror = err => {
      console.error('SSE error', err);
      es.close();
    };

    return () => es.close();
  }, []);

  // 더미 사용자 클릭 → 중계 서버에 전송
  const sendAlert = async user => {
    try {
      console.log('[Frontend → POST /mcs/alert/send]', user);
      setLoadingUsers(prev => new Set([...prev, user.id]));
      
      await fetch('http://172.30.1.28:3003/mcs/alert/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(user)
      });
    } catch (err) {
      console.error(err);
      setLoadingUsers(prev => {
        const newSet = new Set(prev);
        newSet.delete(user.id);
        return newSet;
      });
    }
  };

  const getUserStatus = (userId) => {
    if (loadingUsers.has(userId)) return 'loading';
    if (responses[userId]) return responses[userId];
    return 'idle';
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'loading': return '확인 중...';
      case 'ok': return '안전';
      case 'emergency': return '위험';
      default: return '';
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>2단계 사용자 모니터링 시스템</h1>
      </header>

      <div className="users-list">
        {DUMMY_USERS.map(user => {
          const status = getUserStatus(user.id);
          
          return (
            <div key={user.id} className="user-row">
              <div className="user-info">
                <span className="user-name">{user.name}</span>
                <span className="user-id">({user.id})</span>
                {status !== 'idle' && (
                  <span className={`status ${status}`}>
                    {getStatusText(status)}
                  </span>
                )}
              </div>
              
              <div className="user-actions">
                <button 
                  className="alert-button"
                  onClick={() => sendAlert(user)}
                  disabled={status === 'loading'}
                >
                  {status === 'loading' ? (
                    <>
                      <div className="spinner"></div>
                      확인 중...
                    </>
                  ) : (
                    '알림 전송'
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default App;