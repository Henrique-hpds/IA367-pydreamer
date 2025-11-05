# 📊 Como Avaliar Seu Modelo Treinado

## 🎯 Visão Geral

Existem 3 notebooks principais na pasta `results/atari/`:

1. **`export_my_run.ipynb`** ⬅️ **COMECE AQUI!** (recém criado para você)
2. **`figures.ipynb`** - Gera gráficos de comparação
3. **`makegif.ipynb`** - Cria GIFs dos sonhos do agente

---

## 🚀 Passo a Passo

### **Passo 1: Exportar Métricas do Seu Treinamento**

1. Abra o notebook `results/atari/export_my_run.ipynb`
2. Execute todas as células em ordem (Ctrl+Shift+Enter ou "Run All")
3. O notebook vai:
   - Listar todos os seus runs do MLflow
   - Selecionar automaticamente o run mais recente
   - Exportar as métricas para `runs/atari_pong_myrun_0.csv`

**Resultado:** Arquivo CSV com métricas do seu treinamento salvo em `runs/`

---

### **Passo 2: Gerar Gráficos de Performance**

1. Abra o notebook `results/atari/figures.ipynb`
2. Execute todas as células
3. O notebook vai:
   - Ler todos os CSVs da pasta `runs/` (incluindo o seu!)
   - Comparar com baselines do DreamerV2
   - Gerar gráficos mostrando:
     - Retorno médio vs passos do ambiente
     - Banda de desvio padrão
     - Comparação com DreamerV2
   - Salvar figuras em `figures/atari_pong.png`

**Resultado:** Gráficos de comparação salvos em `figures/`

---

### **Passo 3 (Opcional): Visualizar "Sonhos" do Agente**

O `makegif.ipynb` cria GIFs mostrando o que o modelo imagina (predictions do world model).

**ATENÇÃO:** Este precisa de ajustes:
- Você precisa ter artefatos `d2_wm_dream` salvos no MLflow
- Precisa editar o notebook com o seu `run_id` específico

---

## 📋 Métricas Disponíveis

O notebook exporta automaticamente:

| Métrica | Descrição |
|---------|-----------|
| `return` | Retorno médio dos episódios (recompensa total) |
| `agent_steps` | Número de passos do agente coletados |
| `env_steps` | Passos do ambiente (agent_steps × 4 para Atari) |

---

## 🎨 Exemplo de Uso

```bash
cd results/atari

# 1. Execute o export (via Jupyter ou VS Code)
# Abre export_my_run.ipynb e roda todas as células

# 2. Gere os gráficos
# Abre figures.ipynb e roda todas as células

# 3. Veja os resultados
ls -lh figures/atari_pong.png
```

---

## 📊 Interpretando os Resultados

### **Gráfico de Pong:**
- **Eixo X:** Passos do ambiente (em milhões)
- **Eixo Y:** Retorno médio (-21 a +21)
- **Curva:** Performance do seu modelo
- **Área sombreada:** Desvio padrão
- **Comparação:** Linha do DreamerV2 baseline

### **O que é um bom resultado para Pong?**
- Retorno de **+21**: Perfeito! Ganhando sempre
- Retorno de **0**: Empatando
- Retorno de **-21**: Perdendo sempre

Para Pong, você deve ver melhoria rápida (geralmente em < 1M passos).

---

## 🔧 Personalizações

### **Mudar o run analisado:**

No `export_my_run.ipynb`, célula 3:
```python
# Escolher run específico pelo índice
selected_run = runs.iloc[0]  # Mais recente
selected_run = runs.iloc[1]  # Segundo mais recente
# etc.
```

### **Mudar nome do arquivo de saída:**

Na célula 5:
```python
run_name = f"meu_experimento_pong_1"  # Nome personalizado
```

---

## 🆘 Problemas Comuns

### **"Nenhuma métrica encontrada"**
- Verifique se o treinamento já começou a salvar métricas
- Métricas são salvas a cada `log_interval` (default: 100 passos)

### **"Arquivo CSV vazio"**
- O treinamento pode não ter progredido o suficiente
- Espere alguns minutos e tente novamente

### **Gráfico não aparece no figures.ipynb**
- Certifique-se de ter instalado: `pip install holoviews matplotlib`
- O arquivo CSV precisa estar em `runs/` com padrão `atari_*.csv`

---

## 📚 Arquivos Criados

Após executar os notebooks:

```
results/atari/
├── runs/
│   └── atari_pong_myrun_0.csv     ← Suas métricas
├── figures/
│   └── atari_pong.png             ← Gráfico de comparação
├── export_my_run.ipynb            ← Notebook de exportação
├── figures.ipynb                  ← Notebook de visualização
└── README_EVALUATION.md           ← Este arquivo
```

---

## 🎯 Dicas Finais

1. **Execute export_my_run.ipynb periodicamente** durante o treinamento para ver progresso
2. **Os gráficos atualizam automaticamente** quando você re-executa figures.ipynb
3. **Compare com baselines** para ver se está no caminho certo
4. **Para Pong, espere ~200K env steps** para ver resultados significativos

---

Boa sorte com seu treinamento! 🚀
