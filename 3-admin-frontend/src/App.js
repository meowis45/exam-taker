import React, { useEffect, useMemo, useState } from 'react';
import { MathJax, MathJaxContext } from 'better-react-mathjax';
import './App.css';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const PAGE_SIZE = 25; // MathJax typesetting hundreds of rows at once is slow
const LEVELS = ['Easy', 'Medium', 'Hard'];
const KEYS = ['A', 'B', 'C', 'D'];

function AdminDashboard() {
    const [questions, setQuestions] = useState([]);
    const [status, setStatus] = useState('loading'); // loading | ready | error
    const [error, setError] = useState('');
    const [subject, setSubject] = useState('All');
    const [complexity, setComplexity] = useState('All');
    const [onlyMissingKey, setOnlyMissingKey] = useState(false);
    const [page, setPage] = useState(0);

    useEffect(() => {
        fetch(`${API}/api/admin/questions`)
            .then(res => {
                if (!res.ok) {
                    return res.json().catch(() => ({})).then(body => {
                        throw new Error(body.detail || `Server returned ${res.status}`);
                    });
                }
                return res.json();
            })
            .then(data => {
                setQuestions(data.questions ?? []);
                setStatus('ready');
            })
            .catch(err => {
                setError(err.message);
                setStatus('error');
            });
    }, []);

    const subjects = useMemo(
        () => ['All', ...new Set(questions.map(q => q.subject))],
        [questions]
    );

    const filtered = useMemo(
        () => questions.filter(q =>
            (subject === 'All' || q.subject === subject) &&
            (complexity === 'All' || q.complexity === complexity) &&
            (!onlyMissingKey || !q.correct_option)
        ),
        [questions, subject, complexity, onlyMissingKey]
    );

    const missingKeys = questions.filter(q => !q.correct_option).length;
    const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
    const safePage = Math.min(page, pageCount - 1);
    const visible = filtered.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);

    const onFilter = setter => e => {
        setter(e.target.value);
        setPage(0);
    };

    // Optimistic update; rolls back if the server rejects it.
    const updateQuestion = async (id, patch) => {
        const previous = questions;
        setQuestions(qs => qs.map(q => (q.id === id ? { ...q, ...patch } : q)));
        try {
            const res = await fetch(`${API}/api/admin/questions/${id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patch),
            });
            if (!res.ok) throw new Error(`Server returned ${res.status}`);
        } catch (err) {
            setQuestions(previous);
            alert(`Could not save change: ${err.message}`);
        }
    };

    if (status === 'loading') return <div className="admin-container"><p>Loading questions…</p></div>;

    if (status === 'error') {
        return (
            <div className="admin-container">
                <h1>Database Admin Dashboard</h1>
                <p className="error">Could not load questions: {error}</p>
                <p>Is the backend running (<code>uvicorn main:app --reload</code> in <code>backend/</code>) and has <code>seed_db.py</code> been run?</p>
            </div>
        );
    }

    return (
        <MathJaxContext>
            <div className="admin-container">
                <h1>Database Admin Dashboard</h1>

                <div className="filters">
                    <label>Subject: </label>
                    <select value={subject} onChange={onFilter(setSubject)}>
                        {subjects.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>

                    <label>Complexity: </label>
                    <select value={complexity} onChange={onFilter(setComplexity)}>
                        <option value="All">All</option>
                        {LEVELS.map(l => <option key={l} value={l}>{l}</option>)}
                    </select>

                    <label className="inline">
                        <input
                            type="checkbox"
                            checked={onlyMissingKey}
                            onChange={e => { setOnlyMissingKey(e.target.checked); setPage(0); }}
                        /> Missing answer key only
                    </label>

                    <span className="totals">
                        Showing {filtered.length} of {questions.length} · {missingKeys} without answer key
                    </span>
                </div>

                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Subject</th>
                            <th>Chapter</th>
                            <th>Complexity</th>
                            <th>Question</th>
                            <th>Options</th>
                            <th>Answer</th>
                        </tr>
                    </thead>
                    <tbody>
                        {visible.map(q => (
                            <tr key={q.id}>
                                <td>{q.subject}</td>
                                <td>{q.chapter}</td>
                                <td>
                                    <select
                                        className={`badge ${(q.complexity ?? 'unknown').toLowerCase()}`}
                                        value={q.complexity ?? ''}
                                        onChange={e => updateQuestion(q.id, { complexity: e.target.value })}
                                    >
                                        {!q.complexity && <option value="">—</option>}
                                        {LEVELS.map(l => <option key={l} value={l}>{l}</option>)}
                                    </select>
                                </td>
                                <td>
                                    {/* MathJax renders LaTeX like \( x^2 \) */}
                                    <MathJax>{q.question_text ?? ''}</MathJax>
                                </td>
                                <td>
                                    <ul>
                                        <li>A: <MathJax inline>{q.option_a ?? ''}</MathJax></li>
                                        <li>B: <MathJax inline>{q.option_b ?? ''}</MathJax></li>
                                        <li>C: <MathJax inline>{q.option_c ?? ''}</MathJax></li>
                                        <li>D: <MathJax inline>{q.option_d ?? ''}</MathJax></li>
                                    </ul>
                                </td>
                                <td>
                                    <select
                                        className={q.correct_option ? '' : 'needs-key'}
                                        value={q.correct_option ?? ''}
                                        onChange={e => updateQuestion(q.id, { correct_option: e.target.value })}
                                    >
                                        {!q.correct_option && <option value="">Set…</option>}
                                        {KEYS.map(k => <option key={k} value={k}>{k}</option>)}
                                    </select>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>

                <div className="pager">
                    <button onClick={() => setPage(safePage - 1)} disabled={safePage === 0}>← Prev</button>
                    <span>Page {safePage + 1} of {pageCount}</span>
                    <button onClick={() => setPage(safePage + 1)} disabled={safePage >= pageCount - 1}>Next →</button>
                </div>
            </div>
        </MathJaxContext>
    );
}

export default AdminDashboard;
