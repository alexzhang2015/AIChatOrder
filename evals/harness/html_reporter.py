"""
HTML 报告生成器

生成美观的 HTML 格式评估报告，支持:
- 响应式设计
- 交互式图表
- 详细的失败分析
- 可折叠的试验详情
"""

import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from evals.harness.models import EvalResult, EvalSuiteResult, Trial, TaskCategory, GraderResult


class HTMLReporter:
    """HTML 报告生成器"""

    def __init__(self, output_dir: str = "evals/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_suite_report(
        self,
        result: EvalSuiteResult,
        filename: Optional[str] = None
    ) -> str:
        """生成套件 HTML 报告"""
        if filename:
            filepath = self.output_dir / filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"report_{result.suite_name}_{timestamp}.html"

        html = self._build_html(result)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return str(filepath)

    def _build_html(self, result: EvalSuiteResult) -> str:
        """构建完整 HTML"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>评估报告: {result.suite_name}</title>
    {self._get_styles()}
</head>
<body>
    <div class="container">
        {self._build_header(result)}
        {self._build_summary(result)}
        {self._build_category_stats(result)}
        {self._build_task_details(result)}
        {self._build_footer(result)}
    </div>
    {self._get_scripts()}
</body>
</html>"""

    def _get_styles(self) -> str:
        """CSS 样式"""
        return """
    <style>
        :root {
            --primary: #4F46E5;
            --success: #10B981;
            --danger: #EF4444;
            --warning: #F59E0B;
            --gray-50: #F9FAFB;
            --gray-100: #F3F4F6;
            --gray-200: #E5E7EB;
            --gray-300: #D1D5DB;
            --gray-600: #4B5563;
            --gray-800: #1F2937;
            --gray-900: #111827;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
            color: var(--gray-800);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 1rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            overflow: hidden;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, #7C3AED 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }

        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .header .subtitle {
            opacity: 0.9;
            font-size: 0.95rem;
        }

        .header .badge {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.5rem 1.5rem;
            background: rgba(255,255,255,0.2);
            border-radius: 2rem;
            font-weight: 600;
        }

        /* Summary Cards */
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            padding: 2rem;
            background: var(--gray-50);
        }

        .card {
            background: white;
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .card .value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
        }

        .card .label {
            color: var(--gray-600);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }

        .card.success .value { color: var(--success); }
        .card.danger .value { color: var(--danger); }
        .card.warning .value { color: var(--warning); }

        /* Progress Ring */
        .progress-ring {
            width: 120px;
            height: 120px;
            margin: 0 auto 1rem;
        }

        .progress-ring circle {
            fill: none;
            stroke-width: 8;
        }

        .progress-ring .bg {
            stroke: var(--gray-200);
        }

        .progress-ring .progress {
            stroke: var(--success);
            stroke-linecap: round;
            transform: rotate(-90deg);
            transform-origin: 50% 50%;
            transition: stroke-dashoffset 0.5s ease;
        }

        .progress-ring .value {
            font-size: 1.5rem;
            font-weight: 700;
        }

        /* Category Stats */
        .category-section {
            padding: 2rem;
            border-bottom: 1px solid var(--gray-200);
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--gray-800);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .category-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }

        .category-card {
            background: var(--gray-50);
            border-radius: 0.5rem;
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .category-name {
            font-weight: 500;
            text-transform: capitalize;
        }

        .category-stats {
            text-align: right;
        }

        .category-rate {
            font-size: 1.25rem;
            font-weight: 600;
        }

        .category-count {
            font-size: 0.85rem;
            color: var(--gray-600);
        }

        /* Task List */
        .tasks-section {
            padding: 2rem;
        }

        .task-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .task-item {
            border: 1px solid var(--gray-200);
            border-radius: 0.75rem;
            overflow: hidden;
            transition: box-shadow 0.2s;
        }

        .task-item:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .task-header {
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            background: var(--gray-50);
        }

        .task-header:hover {
            background: var(--gray-100);
        }

        .task-info {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .task-status {
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        .task-status.pass {
            background: #D1FAE5;
            color: var(--success);
        }

        .task-status.fail {
            background: #FEE2E2;
            color: var(--danger);
        }

        .task-name {
            font-weight: 500;
        }

        .task-id {
            font-size: 0.85rem;
            color: var(--gray-600);
        }

        .task-metrics {
            display: flex;
            gap: 1.5rem;
        }

        .metric {
            text-align: center;
        }

        .metric-value {
            font-weight: 600;
            font-size: 1.1rem;
        }

        .metric-label {
            font-size: 0.75rem;
            color: var(--gray-600);
        }

        .task-details {
            display: none;
            padding: 1.5rem;
            background: white;
            border-top: 1px solid var(--gray-200);
        }

        .task-item.expanded .task-details {
            display: block;
        }

        .task-item.expanded .expand-icon {
            transform: rotate(180deg);
        }

        .expand-icon {
            transition: transform 0.2s;
            color: var(--gray-600);
        }

        /* Trial Details */
        .trials-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .trial-card {
            background: var(--gray-50);
            border-radius: 0.5rem;
            padding: 1rem;
        }

        .trial-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .trial-number {
            font-weight: 500;
        }

        .trial-status {
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .trial-status.pass {
            background: #D1FAE5;
            color: #065F46;
        }

        .trial-status.fail {
            background: #FEE2E2;
            color: #991B1B;
        }

        .grader-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .grader-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
            padding: 0.5rem;
            background: white;
            border-radius: 0.25rem;
        }

        .grader-name {
            color: var(--gray-600);
        }

        .grader-score {
            font-weight: 500;
        }

        /* Failures Section */
        .failures-section {
            margin-top: 1rem;
        }

        .failures-title {
            font-weight: 500;
            color: var(--danger);
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .failure-item {
            background: #FEF2F2;
            border-left: 3px solid var(--danger);
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 0 0.25rem 0.25rem 0;
            font-size: 0.9rem;
        }

        .failure-type {
            font-weight: 500;
            color: var(--danger);
        }

        .failure-detail {
            color: var(--gray-600);
            margin-top: 0.25rem;
            font-family: monospace;
            font-size: 0.85rem;
        }

        /* Footer */
        .footer {
            padding: 1.5rem 2rem;
            background: var(--gray-50);
            text-align: center;
            color: var(--gray-600);
            font-size: 0.9rem;
        }

        .footer a {
            color: var(--primary);
            text-decoration: none;
        }

        /* Responsive */
        @media (max-width: 768px) {
            body {
                padding: 1rem;
            }

            .header h1 {
                font-size: 1.5rem;
            }

            .summary {
                grid-template-columns: repeat(2, 1fr);
                padding: 1rem;
                gap: 1rem;
            }

            .card .value {
                font-size: 1.75rem;
            }

            .task-metrics {
                display: none;
            }
        }
    </style>"""

    def _get_scripts(self) -> str:
        """JavaScript 交互"""
        return """
    <script>
        // 展开/折叠任务详情
        document.querySelectorAll('.task-header').forEach(header => {
            header.addEventListener('click', () => {
                const item = header.parentElement;
                item.classList.toggle('expanded');
            });
        });

        // 进度环动画
        document.querySelectorAll('.progress-ring .progress').forEach(circle => {
            const percent = parseFloat(circle.dataset.percent);
            const circumference = 2 * Math.PI * 52;
            const offset = circumference - (percent / 100) * circumference;
            circle.style.strokeDasharray = circumference;
            circle.style.strokeDashoffset = offset;
        });
    </script>"""

    def _build_header(self, result: EvalSuiteResult) -> str:
        """构建头部"""
        status = "✅ 通过" if result.overall_pass_rate >= 0.95 else "⚠️ 需关注" if result.overall_pass_rate >= 0.8 else "❌ 失败"
        return f"""
        <div class="header">
            <h1>🧪 AI 点单系统评估报告</h1>
            <p class="subtitle">套件: {result.suite_name} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <span class="badge">{status}</span>
        </div>"""

    def _build_summary(self, result: EvalSuiteResult) -> str:
        """构建概览卡片"""
        pass_rate = result.overall_pass_rate * 100
        duration = result.total_duration_ms / 1000

        # 确定进度环颜色
        ring_color = "#10B981" if pass_rate >= 90 else "#F59E0B" if pass_rate >= 70 else "#EF4444"

        return f"""
        <div class="summary">
            <div class="card">
                <svg class="progress-ring" viewBox="0 0 120 120">
                    <circle class="bg" cx="60" cy="60" r="52"/>
                    <circle class="progress" cx="60" cy="60" r="52"
                            data-percent="{pass_rate}"
                            style="stroke: {ring_color}"/>
                    <text x="60" y="65" text-anchor="middle" class="value">{pass_rate:.0f}%</text>
                </svg>
                <div class="label">总体通过率</div>
            </div>
            <div class="card success">
                <div class="value">{result.tasks_passed}</div>
                <div class="label">通过任务</div>
            </div>
            <div class="card danger">
                <div class="value">{result.tasks_failed}</div>
                <div class="label">失败任务</div>
            </div>
            <div class="card">
                <div class="value">{result.tasks_total}</div>
                <div class="label">总任务数</div>
            </div>
            <div class="card warning">
                <div class="value">{duration:.1f}s</div>
                <div class="label">总耗时</div>
            </div>
        </div>"""

    def _build_category_stats(self, result: EvalSuiteResult) -> str:
        """构建类别统计"""
        if not result.by_category:
            return ""

        cards = []
        for category, stats in result.by_category.items():
            rate = stats['pass_rate'] * 100
            color = "var(--success)" if rate >= 90 else "var(--warning)" if rate >= 70 else "var(--danger)"
            cards.append(f"""
                <div class="category-card">
                    <div class="category-name">📁 {category}</div>
                    <div class="category-stats">
                        <div class="category-rate" style="color: {color}">{rate:.0f}%</div>
                        <div class="category-count">{stats['passed']}/{stats['total']} 通过</div>
                    </div>
                </div>""")

        return f"""
        <div class="category-section">
            <h2 class="section-title">📊 按类别统计</h2>
            <div class="category-grid">
                {''.join(cards)}
            </div>
        </div>"""

    def _build_task_details(self, result: EvalSuiteResult) -> str:
        """构建任务详情"""
        task_items = []

        for task_result in result.results:
            passed = task_result.pass_all >= 1.0
            status_class = "pass" if passed else "fail"
            status_icon = "✓" if passed else "✗"

            # 构建试验卡片
            trial_cards = []
            all_failures = []

            for trial in task_result.trials:
                grader_items = []
                for grader_type, grader_result in trial.grader_results.items():
                    if isinstance(grader_result, GraderResult):
                        score = grader_result.score
                        g_passed = grader_result.passed
                        failures = grader_result.failures
                    else:
                        score = grader_result.get('score', 0)
                        g_passed = grader_result.get('passed', False)
                        failures = grader_result.get('failures', [])

                    g_color = "var(--success)" if g_passed else "var(--danger)"
                    grader_items.append(f"""
                        <div class="grader-item">
                            <span class="grader-name">{grader_type}</span>
                            <span class="grader-score" style="color: {g_color}">{score:.2f}</span>
                        </div>""")

                    # 收集失败信息
                    for f in failures[:3]:
                        all_failures.append({
                            'type': f.get('type', 'unknown'),
                            'input': str(f.get('input', ''))[:100],
                            'detail': f.get('expected', f.get('reasons', ''))
                        })

                trial_status = "pass" if trial.passed else "fail"
                trial_cards.append(f"""
                    <div class="trial-card">
                        <div class="trial-header">
                            <span class="trial-number">Trial {trial.trial_number + 1}</span>
                            <span class="trial-status {trial_status}">{'PASS' if trial.passed else 'FAIL'}</span>
                        </div>
                        <div class="grader-list">
                            {''.join(grader_items)}
                        </div>
                    </div>""")

            # 构建失败详情
            failures_html = ""
            if all_failures:
                failure_items = []
                for f in all_failures[:5]:
                    failure_items.append(f"""
                        <div class="failure-item">
                            <div class="failure-type">{f['type']}</div>
                            <div class="failure-detail">{f['input']}</div>
                        </div>""")
                failures_html = f"""
                    <div class="failures-section">
                        <div class="failures-title">⚠️ 失败详情</div>
                        {''.join(failure_items)}
                    </div>"""

            task_items.append(f"""
                <div class="task-item">
                    <div class="task-header">
                        <div class="task-info">
                            <div class="task-status {status_class}">{status_icon}</div>
                            <div>
                                <div class="task-name">{task_result.task_name}</div>
                                <div class="task-id">{task_result.task_id}</div>
                            </div>
                        </div>
                        <div class="task-metrics">
                            <div class="metric">
                                <div class="metric-value">{task_result.pass_at_1:.0%}</div>
                                <div class="metric-label">pass@1</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">{task_result.pass_at_k:.0%}</div>
                                <div class="metric-label">pass@k</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">{task_result.pass_all:.0%}</div>
                                <div class="metric-label">pass_all</div>
                            </div>
                        </div>
                        <span class="expand-icon">▼</span>
                    </div>
                    <div class="task-details">
                        <div class="trials-grid">
                            {''.join(trial_cards)}
                        </div>
                        {failures_html}
                    </div>
                </div>""")

        return f"""
        <div class="tasks-section">
            <h2 class="section-title">📋 任务详情</h2>
            <div class="task-list">
                {''.join(task_items)}
            </div>
        </div>"""

    def _build_footer(self, result: EvalSuiteResult) -> str:
        """构建页脚"""
        return f"""
        <div class="footer">
            <p>基于 <a href="https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents" target="_blank">Anthropic AI Agent Evals</a> 方法论</p>
            <p>开始时间: {result.started_at} | 完成时间: {result.completed_at or 'N/A'}</p>
        </div>"""
