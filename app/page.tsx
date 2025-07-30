"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Award,
  BarChart3,
  Brain,
  Calendar,
  ChevronRight,
  Code,
  Database,
  Github,
  Linkedin,
  Mail,
  Moon,
  Play,
  Star,
  Sun,
} from "lucide-react"
import { useState } from "react"

interface CellOutput {
  type: "dataframe" | "text" | "plot" | "ml"
  content: any
}

interface NotebookCell {
  id: string
  code: string
  output?: CellOutput
  executed: boolean
}

export default function Portfolio() {
  const [executedCells, setExecutedCells] = useState<Set<string>>(new Set())
  const [isDark, setIsDark] = useState(false)

  const personalData = {
    basic_info: [
      { field: "name", value: "Fnine Jasser", type: "string" },
      { field: "title", value: "ML Engineer", type: "string" },
      { field: "experience_years", value: 1, type: "int64" },
      { field: "education", value: "Master's in AI, Université Côte d'Azur, Nice France", type: "string" },
    ],
    skills: [
      { skill: "Python", proficiency: 95, category: "Programming" },
      { skill: "Machine Learning", proficiency: 92, category: "AI/ML" },
      { skill: "Deep Learning", proficiency: 88, category: "AI/ML" },
      { skill: "SQL", proficiency: 90, category: "Database" },
      { skill: "Java", proficiency: 82, category: "Programming" },
      { skill: "TensorFlow", proficiency: 87, category: "Framework" },
      { skill: "PyTorch", proficiency: 83, category: "Framework" },
      { skill: "Pandas", proficiency: 98, category: "Data Analysis" },
    ],
    projects: [
      {
        name: "Economic News Intelligence And Sentiment Analysis",
        impact: "Real-time market insights with 92% accuracy",
        tech_stack: "NLP, LLM, Dash, Plotly, Python",
        year: 2025,
      },
      {
        name: "Stock Portfolio Analysis using Deep Learning",
        impact: "Optimized portfolio performance by 25%",
        tech_stack: "Deep Learning, PyTorch, Financial APIs",
        year: 2025,
      },
      {
        name: "Multimodal Search Engine",
        impact: "Enhanced search relevance by 40%",
        tech_stack: "Deep Learning, PyTorch, AudioCLIP",
        year: 2025,
      },
    ],
  }

  const notebookCells: NotebookCell[] = [
    {
      id: "cell-1",
      code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load personal data
personal_df = pd.DataFrame({
    'field': ['name', 'title', 'experience_years', 'education'],
    'value': ['Fnine Jasser', 'ML Engineer', 1, "Master's in AI, Université Côte d'Azur, Nice France"],
    'type': ['string', 'string', 'int64', 'string']
})

print("Personal Information Dataset:")
personal_df`,
      executed: false,
    },
    {
      id: "cell-2",
      code: `# Analyze skills dataset
skills_df = pd.DataFrame({
    'skill': ['Python', 'Machine Learning', 'Deep Learning', 'SQL', 'JavaScript', 'TensorFlow', 'PyTorch', 'Pandas'],
    'proficiency': [95, 92, 88, 90, 82, 87, 83, 98],
    'category': ['Programming', 'AI/ML', 'AI/ML', 'Database', 'Programming', 'Framework', 'Framework', 'Data Analysis']
})

print("Skills Analysis:")
print(f"Total skills: {len(skills_df)}")
print(f"Average proficiency: {skills_df['proficiency'].mean():.1f}%")
print(f"Top skill: {skills_df.loc[skills_df['proficiency'].idxmax(), 'skill']}")
skills_df.head()`,
      executed: false,
    },
    {
      id: "cell-3",
      code: `# Visualize skills proficiency
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
skills_df.plot(x='skill', y='proficiency', kind='bar', ax=plt.gca(), color='skyblue')
plt.title('Skills Proficiency')
plt.xticks(rotation=45)
plt.ylabel('Proficiency (%)')

plt.subplot(1, 2, 2)
category_avg = skills_df.groupby('category')['proficiency'].mean()
plt.pie(category_avg.values, labels=category_avg.index, autopct='%1.1f%%', colors=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
plt.title('Skills by Category')

plt.tight_layout()
plt.show()`,
      executed: false,
    },
    {
      id: "cell-4",
      code: `# Simple ML model demonstration
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create synthetic project success data
np.random.seed(42)
project_data = pd.DataFrame({
    'team_size': np.random.randint(3, 12, 50),
    'duration_months': np.random.randint(2, 18, 50),
    'complexity_score': np.random.randint(1, 10, 50),
    'budget_k': np.random.randint(50, 500, 50)
})

# Target: project success score (synthetic)
project_data['success_score'] = (
    project_data['team_size'] * 0.3 + 
    (20 - project_data['duration_months']) * 0.2 + 
    project_data['complexity_score'] * 0.4 + 
    project_data['budget_k'] * 0.001 + 
    np.random.normal(0, 1, 50)
)

# Train model
X = project_data[['team_size', 'duration_months', 'complexity_score', 'budget_k']]
y = project_data['success_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Model R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nFeature Importance:")
feature_importance`,
      executed: false,
    },
    {
      id: "cell-5",
      code: `# Projects timeline and impact analysis
projects_df = pd.DataFrame({
    'project': ['Economic News Intelligence And Sentiment Analysis', 'Stock Portfolio Analysis using Deep Learning', 'Multimodal Search Engine'],
    'impact': ['Real-time market insights with 92% accuracy', 'Optimized portfolio performance by 25%', 'Enhanced search relevance by 40%'],
    'tech_stack': ['NLP, LLM, Dash, Plotly, Python', 'Deep Learning, PyTorch, Financial APIs', 'Deep Learning, PyTorch, AudioCLIP'],
    'year': [2025, 2025, 2025],
    'impact_value': [92, 25, 40]  # Numeric values for visualization
})

plt.figure(figsize=(10, 6))
plt.bar(projects_df['project'], projects_df['impact_value'], color=['#87CEEB', '#F5DEB3', '#98D8E8'])
plt.title('Project Impact Analysis')
plt.ylabel('Impact Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("Project Portfolio Summary:")
print(f"Total projects completed: {len(projects_df)}")
print(f"Average impact: {projects_df['impact_value'].mean():.1f}%")
projects_df[['project', 'impact', 'year']]`,
      executed: false,
    },
  ]

  const executeCell = (cellId: string) => {
    setExecutedCells((prev) => new Set([...prev, cellId]))
  }

  const executeAllCells = () => {
    notebookCells.forEach((cell) => {
      setTimeout(() => executeCell(cell.id), notebookCells.indexOf(cell) * 800)
    })
  }

  const renderDataFrame = (data: any[], columns?: string[]) => {
    if (!data.length) return null

    const cols = columns || Object.keys(data[0])

    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-slate-600">
              <th className="text-left p-2 font-mono text-slate-400"></th>
              {cols.map((col) => (
                <th key={col} className="text-left p-2 font-mono text-slate-300">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx} className="border-b border-slate-700">
                <td className="p-2 font-mono text-slate-500">{idx}</td>
                {cols.map((col) => (
                  <td key={col} className="p-2 font-mono text-slate-200">
                    {typeof row[col] === "number" ? row[col] : `"${row[col]}"`}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  const renderSyntaxHighlightedCode = (code: string) => {
    const lines = code.split("\n")
    return (
      <div className="font-mono text-sm">
        {lines.map((line, idx) => (
          <div key={idx} className="leading-relaxed">
            {highlightPythonSyntax(line)}
          </div>
        ))}
      </div>
    )
  }

  const highlightPythonSyntax = (line: string) => {
    // Handle comments first
    if (line.trim().startsWith("#")) {
      return <span className="text-slate-500">{line}</span>
    }

    const tokens = []
    let currentToken = ""
    let i = 0

    const keywords = [
      "import",
      "from",
      "def",
      "class",
      "if",
      "else",
      "elif",
      "for",
      "while",
      "try",
      "except",
      "with",
      "as",
      "return",
      "print",
      "True",
      "False",
      "None",
    ]

    while (i < line.length) {
      const char = line[i]

      // Handle strings
      if (char === '"' || char === "'") {
        const quote = char
        let stringContent = quote
        i++
        while (i < line.length && line[i] !== quote) {
          if (line[i] === "\\" && i + 1 < line.length) {
            stringContent += line[i] + line[i + 1]
            i += 2
          } else {
            stringContent += line[i]
            i++
          }
        }
        if (i < line.length) {
          stringContent += line[i] // closing quote
          i++
        }
        tokens.push(
          <span key={tokens.length} className="text-green-600">
            {stringContent}
          </span>,
        )
        continue
      }

      // Handle word characters
      if (/[a-zA-Z_]/.test(char)) {
        currentToken = ""
        while (i < line.length && /[a-zA-Z0-9_]/.test(line[i])) {
          currentToken += line[i]
          i++
        }

        if (keywords.includes(currentToken)) {
          tokens.push(
            <span key={tokens.length} className="text-blue-600">
              {currentToken}
            </span>,
          )
        } else if (i < line.length && line[i] === "(") {
          // Function call
          tokens.push(
            <span key={tokens.length} className="text-purple-600">
              {currentToken}
            </span>,
          )
        } else {
          tokens.push(
            <span key={tokens.length} className="text-slate-300">
              {currentToken}
            </span>,
          )
        }
        continue
      }

      // Handle numbers
      if (/\d/.test(char)) {
        currentToken = ""
        while (i < line.length && /[\d.]/.test(line[i])) {
          currentToken += line[i]
          i++
        }
        tokens.push(
          <span key={tokens.length} className="text-orange-600">
            {currentToken}
          </span>,
        )
        continue
      }

      // Handle other characters (operators, punctuation, etc.)
      tokens.push(
        <span key={tokens.length} className="text-slate-300">
          {char}
        </span>,
      )
      i++
    }

    return <>{tokens}</>
  }

  const getCellOutput = (cellId: string) => {
    if (!executedCells.has(cellId)) return null

    switch (cellId) {
      case "cell-1":
        return (
          <div className="space-y-2">
            <div className="text-slate-300">Personal Information Dataset:</div>
            {renderDataFrame(personalData.basic_info)}
          </div>
        )
      case "cell-2":
        return (
          <div className="space-y-2">
            <div className="text-slate-300">Skills Analysis:</div>
            <div className="font-mono text-sm text-slate-300">
              <div>Total skills: 8</div>
              <div>Average proficiency: 89.4%</div>
              <div>Top skill: Pandas</div>
            </div>
            {renderDataFrame(personalData.skills.slice(0, 5))}
          </div>
        )
      case "cell-3":
        return (
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg shadow-sm">
              <div className="text-center text-gray-600 mb-2">Skills Proficiency & Category Distribution</div>
              <img
                src="/skills.png?height=900&width=600&text=Skills+Proficiency+Bar+Chart+and+Category+Pie+Chart"
                alt="Skills proficiency charts"
                className="w-full h-96 object-contain rounded"
              />
            </div>
          </div>
        )
      case "cell-4":
        const featureData = [
          { feature: "complexity_score", importance: 0.342 },
          { feature: "team_size", importance: 0.298 },
          { feature: "duration_months", importance: 0.201 },
          { feature: "budget_k", importance: 0.159 },
        ]
        return (
          <div className="space-y-2">
            <div className="font-mono text-sm text-slate-300">
              <div>Model R² Score: 0.847</div>
              <div>RMSE: 1.234</div>
            </div>
            <div className="mt-4 text-slate-300">Feature Importance:</div>
            {renderDataFrame(featureData)}
          </div>
        )
      case "cell-5":
        return (
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg shadow-sm">
              <div className="text-center text-gray-600 mb-2">Project Impact Analysis</div>
              <img
                src="/impact.png?height=800&width=500&text=Project+Impact+Bar+Chart+showing+34%+28%+45%+values"
                alt="Project impact visualization"
                className="w-full h-96 object-contain rounded"
              />
            </div>
            <div className="font-mono text-sm text-slate-300">
              <div>Total projects completed: 3</div>
              <div>Average impact: 52.3%</div>
            </div>
            {renderDataFrame(personalData.projects.map((p) => ({ project: p.name, impact: p.impact, year: p.year })))}
          </div>
        )
      default:
        return null
    }
  }

  const themeClasses = {
    background: isDark
      ? "bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900"
      : "bg-gradient-to-br from-blue-50 via-white to-amber-50",
    text: isDark ? "text-white" : "text-gray-900",
    textSecondary: isDark ? "text-slate-300" : "text-gray-600",
    card: isDark
      ? "bg-slate-800/80 backdrop-blur-xl border-slate-700/50"
      : "bg-white/80 backdrop-blur-xl border-blue-200/50 shadow-xl",
    button: isDark
      ? "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
      : "bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800",
    notebook: isDark ? "bg-slate-900/90" : "bg-gray-900", // Notebook background is always dark
  }

  return (
    <div className={`min-h-screen transition-all duration-500 ${themeClasses.background} relative overflow-hidden`}>
      {/* Floating Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className={`absolute top-20 right-20 w-64 h-64 rounded-full opacity-20 animate-pulse ${
            isDark ? "bg-blue-500" : "bg-blue-300"
          } blur-3xl`}
        ></div>
        <div
          className={`absolute bottom-20 left-20 w-64 h-64 rounded-full opacity-20 animate-pulse ${
            isDark ? "bg-cyan-500" : "bg-amber-300"
          } blur-3xl animation-delay-2000`}
        ></div>
        <div
          className={`absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 rounded-full opacity-10 animate-pulse ${
            isDark ? "bg-purple-500" : "bg-blue-200"
          } blur-3xl animation-delay-4000`}
        ></div>
      </div>

      {/* Theme Toggle */}
      <div className="fixed top-6 right-6 z-50">
        <Button
          onClick={() => setIsDark(!isDark)}
          variant="outline"
          size="sm"
          className={`${
            isDark
              ? "bg-slate-800/80 border-slate-600 text-white hover:bg-slate-700"
              : "bg-white/80 border-blue-200 text-gray-700 hover:bg-blue-50"
          } backdrop-blur-xl transition-all duration-300`}
        >
          {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </Button>
      </div>

      {/* Hero Section */}
      <section className="relative z-10 pt-20">
        <div className="max-w-7xl mx-auto px-4 py-24">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div
                className={`inline-flex items-center px-4 py-2 rounded-full border transition-all duration-300 ${
                  isDark
                    ? "bg-blue-500/20 border-blue-500/30 text-blue-300"
                    : "bg-blue-100 border-blue-300 text-blue-700"
                }`}
              >
                <Star className="w-4 h-4 mr-2" />
                <span className="text-sm font-medium">Available for new opportunities</span>
              </div>

              <div>
                <h1
                  className={`text-6xl md:text-7xl font-bold mb-4 ${themeClasses.text} transition-colors duration-300`}
                >
                  Fnine Jasser
                </h1>
                <p className={`text-xl md:text-2xl mb-6 ${themeClasses.textSecondary} transition-colors duration-300`}>
                  ML Engineer & AI Specialist
                </p>
              </div>

              <div className={`flex items-center gap-6 ${themeClasses.textSecondary} transition-colors duration-300`}>
                <div className="flex items-center gap-2">
                  <Calendar className="w-5 h-5" />
                  <span>1 Year Experience</span>
                </div>
                <div className="flex items-center gap-2">
                  <Award className="w-5 h-5" />
                  <span>Master's in AI, Université Côte d'Azur, Nice France</span>
                </div>
              </div>

              <div className="flex gap-4">
                <a
                  href="https://github.com/JsFn99"
                  target="_blank"
                  rel="noopener noreferrer"
                  tabIndex={-1}
                >
                  <Button className={`${themeClasses.button} text-white shadow-lg transition-all duration-300`}>
                    <Github className="w-5 h-5 mr-2" />
                    GitHub
                  </Button>
                </a>
                <a
                  href="https://www.linkedin.com/in/jasser-fnine/"
                  target="_blank"
                  rel="noopener noreferrer"
                  tabIndex={-1}
                >
                  <Button
                    variant="outline"
                    className={`transition-all duration-300 ${
                      isDark
                        ? "border-blue-500/50 text-blue-300 hover:bg-blue-500/10"
                        : "border-blue-500 text-blue-600 hover:bg-blue-50"
                    } bg-transparent backdrop-blur`}
                  >
                    <Linkedin className="w-5 h-5 mr-2" />
                    LinkedIn
                  </Button>
                </a>
                <a
                  href="mailto:fninejasser@hotmail.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  tabIndex={-1}
                >
                  <Button
                    variant="outline"
                    className={`transition-all duration-300 ${
                      isDark
                        ? "border-amber-500/50 text-amber-300 hover:bg-amber-500/10"
                        : "border-amber-500 text-amber-600 hover:bg-amber-50"
                    } bg-transparent backdrop-blur`}
                  >
                    <Mail className="w-5 h-5 mr-2" />
                    Contact
                  </Button>
                </a>
              </div>
            </div>

            <div className="flex justify-center">
              <div className="relative group">
                <div className="w-80 h-80 rounded-2xl bg-gradient-to-br from-blue-400 via-blue-500 to-amber-400 p-1 shadow-2xl group-hover:shadow-blue-500/25 transition-all duration-300">
                  <div
                    className={`w-full h-full rounded-2xl p-2 ${
                      isDark ? "bg-slate-900" : "bg-white"
                    } transition-colors duration-300`}
                  >
                    <img
                      src="/profil.jpg?height=400&width=400&text=Professional+Portrait"
                      alt="Fnine Jasser"
                      className="w-full h-full rounded-xl object-cover"
                    />
                  </div>
                </div>
                <div className="absolute -bottom-4 -right-4 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-full p-4 shadow-lg group-hover:scale-110 transition-transform duration-300">
                  <Brain className="w-8 h-8 text-white" />
                </div>
                <div className="absolute -top-4 -left-4 bg-gradient-to-r from-amber-500 to-orange-500 rounded-full p-3 shadow-lg group-hover:scale-110 transition-transform duration-300">
                  <BarChart3 className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section className="py-20 px-4 relative z-10">
        <div className="max-w-4xl mx-auto">
          <h2 className={`text-4xl font-bold text-center mb-8 ${themeClasses.text} transition-colors duration-300`}>
            About Me
          </h2>
          <Card className={`${themeClasses.card} transition-all duration-300 hover:shadow-2xl hover:-translate-y-1`}>
            <CardContent className="p-8">
              <p className={`${themeClasses.textSecondary} text-lg leading-relaxed transition-colors duration-300`}>
                AI and Data Science Engineer with a solid background in Applied Artificial Intelligence and Computer 
                Engineering. Passionate about building intelligent, data-driven systems with real-world impact. Experienced
                in developing machine learning pipelines, NLP solutions, and full-stack applications. Strong expertise in AI
                model development, data engineering, and problem-solving, with a hands-on approach to delivering
                innovative, production-ready solutions.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Skills Section */}
      <section className="py-20 px-4 relative z-10">
        <div className="max-w-6xl mx-auto">
          <h2 className={`text-4xl font-bold text-center mb-12 ${themeClasses.text} transition-colors duration-300`}>
            Technical Expertise
          </h2>
          <div className="grid md:grid-cols-4 gap-8">
            {" "}
            {/* Changed to 4 columns */}
            <Card
              className={`${themeClasses.card} transition-all duration-300 hover:shadow-2xl hover:-translate-y-2 group`}
            >
              <CardHeader>
                <CardTitle className={`${themeClasses.text} flex items-center transition-colors duration-300`}>
                  <div className="p-2 rounded-lg bg-gradient-to-r from-blue-600 to-cyan-600 mr-3 group-hover:scale-110 transition-transform duration-300">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  AI & Machine Learning
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {["TensorFlow", "PyTorch", "Scikit-learn", "Deep Learning"].map((skill) => (
                    <Badge
                      key={skill}
                      className={`transition-all duration-300 ${
                        isDark
                          ? "bg-blue-500/20 text-blue-300 border-blue-500/30 hover:bg-blue-500/30"
                          : "bg-blue-100 text-blue-700 border-blue-300 hover:bg-blue-200"
                      }`}
                    >
                      {skill}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
            <Card
              className={`${themeClasses.card} transition-all duration-300 hover:shadow-2xl hover:-translate-y-2 group`}
            >
              <CardHeader>
                <CardTitle className={`${themeClasses.text} flex items-center transition-colors duration-300`}>
                  <div className="p-2 rounded-lg bg-gradient-to-r from-amber-600 to-orange-600 mr-3 group-hover:scale-110 transition-transform duration-300">
                    <BarChart3 className="w-6 h-6 text-white" />
                  </div>
                  Data Engineering
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {["Pandas", "Apache Spark", "Kafka", "Airflow"].map((skill) => (
                    <Badge
                      key={skill}
                      className={`transition-all duration-300 ${
                        isDark
                          ? "bg-amber-500/20 text-amber-300 border-amber-500/30 hover:bg-amber-500/30"
                          : "bg-amber-100 text-amber-700 border-amber-300 hover:bg-amber-200"
                      }`}
                    >
                      {skill}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
            <Card
              className={`${themeClasses.card} transition-all duration-300 hover:shadow-2xl hover:-translate-y-2 group`}
            >
              <CardHeader>
                <CardTitle className={`${themeClasses.text} flex items-center transition-colors duration-300`}>
                  <div className="p-2 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 mr-3 group-hover:scale-110 transition-transform duration-300">
                    <Database className="w-6 h-6 text-white" />
                  </div>
                  Infrastructure
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {["AWS", "Docker", "Kubernetes", "MLflow"].map((skill) => (
                    <Badge
                      key={skill}
                      className={`transition-all duration-300 ${
                        isDark
                          ? "bg-purple-500/20 text-purple-300 border-purple-500/30 hover:bg-purple-500/30"
                          : "bg-purple-100 text-purple-700 border-purple-300 hover:bg-purple-200"
                      }`}
                    >
                      {skill}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
            {/* New Dev Tools Card */}
            <Card
              className={`${themeClasses.card} transition-all duration-300 hover:shadow-2xl hover:-translate-y-2 group`}
            >
              <CardHeader>
                <CardTitle className={`${themeClasses.text} flex items-center transition-colors duration-300`}>
                  <div className="p-2 rounded-lg bg-gradient-to-r from-green-600 to-emerald-600 mr-3 group-hover:scale-110 transition-transform duration-300">
                    <Code className="w-6 h-6 text-white" />
                  </div>
                  Dev Tools
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {["Git", "VS Code", "Jupyter", "Docker Compose", "DVC"].map((tool) => (
                    <Badge
                      key={tool}
                      className={`transition-all duration-300 ${
                        isDark
                          ? "bg-green-500/20 text-green-300 border-green-500/30 hover:bg-green-500/30"
                          : "bg-green-100 text-green-700 border-green-300 hover:bg-green-200"
                      }`}
                    >
                      {tool}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Certificates & Achievements Section */}
      <section className="py-20 px-4 relative z-10">
        <div className="max-w-6xl mx-auto">
          <h2 className={`text-4xl font-bold text-center mb-12 ${themeClasses.text} transition-colors duration-300`}>
            Certificates & Achievements
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            {[
              {
                year: "2025",
                title: "Data Science methodology by IBM",
                description:
                  "Comprehensive program covering data science methodologies and best practices for real-world applications.",
                color: "blue",
                logo: "https://yt3.googleusercontent.com/dhVlUr4qzdw97K77mitoVSZk8u3KLl4hWCeiAVNuoqG1W7WmcN86GSIl96Ge1PeauemTwCl5TA=s900-c-k-c0x00ffffff-no-rj",
                institution: "IBM",
              },
              {
                year: "2025",
                title: "Tools for Data Science by IBM",
                description:
                  "In-depth training on essential tools and technologies used in modern data science workflows.",
                color: "blue",
                logo: "https://yt3.googleusercontent.com/dhVlUr4qzdw97K77mitoVSZk8u3KLl4hWCeiAVNuoqG1W7WmcN86GSIl96Ge1PeauemTwCl5TA=s900-c-k-c0x00ffffff-no-rj",
                institution: "IBM",
              },
              {
                year: "2025",
                title: "Big Data with Spark and Hadoop by IBM",
                description:
                  "Advanced certification covering big data processing with Apache Spark and Hadoop ecosystems.",
                color: "blue",
                logo: "https://yt3.googleusercontent.com/dhVlUr4qzdw97K77mitoVSZk8u3KLl4hWCeiAVNuoqG1W7WmcN86GSIl96Ge1PeauemTwCl5TA=s900-c-k-c0x00ffffff-no-rj",
                institution: "IBM",
              },
              {
                year: "2023",
                title: "Specialisation Python for Everybody",
                description: "Comprehensive Python programming specialization covering fundamentals to advanced concepts.",
                color: "amber",
                logo: "https://yt3.googleusercontent.com/ytc/AIdro_nJteQVkAu1W-rpXrFWo7gtxHLfVfhdNnneD_8MCVGdeMc=s900-c-k-c0x00ffffff-no-rj",
                institution: "University of Michigan",
              },
            ].map((cert, index) => (
              <Card
                key={index}
                className={`${themeClasses.card} transition-all duration-300 hover:shadow-2xl hover:-translate-y-1 group`}
              >
                <CardHeader>
                  <CardTitle className={`${themeClasses.text} flex items-center transition-colors duration-300`}>
                    <div className="flex items-center mr-3">
                      <img 
                        src={cert.logo}
                        alt={`${cert.institution} logo`}
                        className="w-8 h-8 object-contain mr-2"
                      />
                      <div
                        className={`px-3 py-1 rounded-full text-white text-sm font-bold bg-gradient-to-r ${
                          cert.color === "blue"
                            ? "from-blue-600 to-cyan-600"
                            : cert.color === "amber"
                              ? "from-amber-600 to-orange-600"
                              : cert.color === "purple"
                                ? "from-purple-600 to-pink-600"
                                : "from-green-600 to-emerald-600"
                        } group-hover:scale-110 transition-transform duration-300`}
                      >
                        {cert.year}
                      </div>
                    </div>
                    <div className="flex-1">
                      {cert.title}
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="mb-2">
                    <span className={`text-sm font-semibold ${themeClasses.textSecondary} transition-colors duration-300`}>
                      {cert.institution}
                    </span>
                  </div>
                  <p className={`${themeClasses.textSecondary} transition-colors duration-300`}>{cert.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Interactive Notebook Section */}
      <section className="py-20 px-4 relative z-10">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className={`text-4xl font-bold mb-4 ${themeClasses.text} transition-colors duration-300`}>
              Interactive Data Profile
            </h2>
            <p className={`${themeClasses.textSecondary} mb-6 transition-colors duration-300`}>
              Explore my professional information through an interactive Jupyter-style notebook
            </p>
            <Button
              onClick={executeAllCells}
              className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white shadow-lg transition-all duration-300"
            >
              <Play className="w-4 h-4 mr-2" />
              Run All Cells
            </Button>
          </div>

          <Card className={`bg-gray-900/90 border-gray-700 shadow-2xl transition-all duration-300`}>
            <CardHeader className="bg-gray-800 border-b border-gray-700">
              <CardTitle className="text-white font-mono text-sm flex items-center">
                <div className="flex space-x-2 mr-4">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                </div>
                fnine_jasser_profile.ipynb
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {notebookCells.map((cell, index) => (
                <div key={cell.id} className="border-b border-gray-700 last:border-b-0">
                  {/* Cell Input */}
                  <div className="flex">
                    <div className="w-20 bg-gray-800 flex items-start justify-center py-4 border-r border-gray-700">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => executeCell(cell.id)}
                        className="text-gray-400 hover:text-white p-1 transition-colors duration-200"
                        disabled={executedCells.has(cell.id)}
                      >
                        {executedCells.has(cell.id) ? (
                          <span className="text-green-400 font-mono text-xs">[{index + 1}]</span>
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                      </Button>
                    </div>
                    <div className="flex-1 p-4 bg-gray-900">{renderSyntaxHighlightedCode(cell.code)}</div>
                  </div>

                  {/* Cell Output */}
                  {executedCells.has(cell.id) && (
                    <div className="flex">
                      <div className="w-20 bg-gray-800 border-r border-gray-700"></div>
                      <div className="flex-1 p-4 bg-gray-850">
                        <div className="font-mono text-sm">{getCellOutput(cell.id)}</div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Projects Section */}
      <section className="py-20 px-4 relative z-10">
        <div className="max-w-6xl mx-auto">
          <h2 className={`text-4xl font-bold text-center mb-12 ${themeClasses.text} transition-colors duration-300`}>
            Featured Projects
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            {personalData.projects.map((project, index) => (
              <Card
                key={index}
                className={`${themeClasses.card} transition-all duration-300 hover:shadow-2xl hover:-translate-y-2 group`}
              >
                <CardHeader>
                  <CardTitle
                    className={`${themeClasses.text} group-hover:text-blue-600 transition-colors duration-300`}
                  >
                    {project.name}
                  </CardTitle>
                  <div className={`text-sm ${themeClasses.textSecondary} transition-colors duration-300`}>
                    {project.year}
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-green-600 mb-4 font-semibold">{project.impact}</p>
                  <p className={`text-sm ${themeClasses.textSecondary} mb-4 transition-colors duration-300`}>
                    {project.tech_stack}
                  </p>
                  <Button
                    variant="ghost"
                    className={`transition-all duration-300 p-0 ${
                      isDark
                        ? "text-blue-400 hover:text-blue-300 hover:bg-blue-500/10"
                        : "text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                    }`}
                  >
                    View Details <ChevronRight className="w-4 h-4 ml-1" />
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="py-20 px-4 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className={`text-4xl font-bold mb-8 ${themeClasses.text} transition-colors duration-300`}>
            Let's Build Something Amazing
          </h2>
          <p className={`${themeClasses.textSecondary} mb-8 transition-colors duration-300`}>
            Ready to discuss your next ML project or explore collaboration opportunities?
          </p>
          <a
            href="mailto:fninejasser@hotmail.com"
            target="_blank"
            rel="noopener noreferrer"
            tabIndex={-1}
          >
            <Button
              size="lg"
              className={`${themeClasses.button} text-white shadow-lg transition-all duration-300 hover:scale-105`}
            >
              <Mail className="w-5 h-5 mr-2" />
              Get In Touch
            </Button>
          </a>
        </div>
      </section>
    </div>
  )
}
