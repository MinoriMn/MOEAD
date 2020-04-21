# MOEAD
## 実行の流れ
- `jupyter notebook`
- pythonファイル書きだし `jupyter nbconvert --to python *.ipynb`

## 構成
- `Main` : メイン処理
- `MOEAD` : MOEA/D 本体

## 実行
- 1000世代実行
- 結果ファイル
 - <a href="https://github.com/MinoriMn/MOEAD/blob/run/gen1000/solution_profit_weight_gen1000.xlsx">`solution_profit_weight_gen1000.xlsx`</a> 最終世代の各個体のprofitとweight
 - <a href="https://github.com/MinoriMn/MOEAD/blob/run/gen1000/solution_x.csv">`solution_x.csv`</a> 最終世代の各個体の遺伝子

<img src="gen1000_graph.png" width="40%">
