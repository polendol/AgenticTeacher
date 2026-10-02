## deberta-trained-pefted-cefr
TEST: {'test_loss': 0.46324828267097473,  
    'test_qwk': 0.7706145974737674, 
    'test_adjacent_acc': 0.9694947569113441, 
    'test_macro_f1': 0.5760069348195234, 
    'test_exact_acc': 0.559580552907531, 
    'test_runtime': 82.6138, 
    'test_samples_per_second': 12.698, 
    'test_steps_per_second': 0.399}

Quadratic Weighted Kappa: agreement between two raters or a predictive model and ground truth on an ordinal scale from 0 to 1, 0.77 = substantial agreement
Adjacent accuracy: an evaluation metric used in ordinal classification where a prediction is considered "correct" if it falls within one rank or level of the true class label, 97% correct or missed by a single level
Macro-F1: the unweighted arithmetic mean of all per-class F1 scores, treating every class equally regardless of its sample size or support, 58% it might be good on some levels but underperform on others
Accuracy: 56%


Confusion matrix  --> better in deberta actually, as well as based on the qwk
    A1  A2   B1   B2   C1  C2
A1   5   2    0    0    0   0
A2   2  30   18    2    0   0
B1   0  37  131   52    5   0
B2   0   7   79  220  105   6
C1   0   0   10   62  140  40
C2   0   0    0    2   33  61

Lots of confusion between B2-C1, B2 level often mistaken for C1

## modernberta-trained-pefted-cefr
TEST: {'test_loss': 0.4720533788204193, 
    'test_qwk': 0.7576356831703336, 
    'test_adjacent_acc': 0.9714013346043852, 
    'test_macro_f1': 0.58880494444519, 
    'test_exact_acc': 0.567206863679695, 
    'test_runtime': 123.8522, 
    'test_samples_per_second': 8.47, 
    'test_steps_per_second': 0.266}

qwk: 76%, slightly lower than deberta
adjacent accuracy: 97%, same as deberta
macro-F1: 59%, almost the same again
accuracy: 57%, 1 % more than deberta

Confusion matrix
    A1  A2   B1   B2   C1  C2
A1   4   3    0    0    0   0
A2   0  26   21    5    0   0
B1   0  19  116   82    7   1
B2   0   0   64  229  115   9
C1   0   0    5   53  159  35
C2   0   0    0    3   32  61