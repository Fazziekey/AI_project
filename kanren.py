# Developer：Fazzie
# Time: 2021/3/2015:06
# File name: kanren.py
# Development environment: Anaconda Python

from kanren import run, eq, membero, var, conde        # kanren一个描述性Python逻辑编程系统
from kanren.core import lall   # lall包用于定义规则

# 等价关系格式一: eq(var(), value) / eq(var(), var())
x = var()                        # 变量声明，kanren的推理基于变量var进行
z = var()
run(0, x, eq(x, z), eq(z, 3))    # 规则求解器，kanren的推理通过run函数进行
                                 # 格式要求为: run(n, var(), rules,[rules, ...])
                                 # 求解指定规则下符合的变量结果
# 等价关系格式二: (eq, var(), value) / (eq, var(), var())
x = var()
z = var()
run(0, x, (eq, x, z), (eq, z, 3))

# 属于关系格式 membero(var(), list / tuple)
x = var()
run(0, x, membero(x, (1, 2, 3)),  # x is a member of (1, 2, 3) #x是（1,2,3）的成员之一
          membero(x, (2, 3, 4)))  # x is a member of (2, 3, 4) #x是（2,3,4）的成员之一


# 逻辑和关系格式 conde((rules, rules))
x = var()
run(0, x, conde((membero(x, (1, 2, 3)), membero(x, (2, 3, 4)))))

# 逻辑或关系格式 conde([rules], [rules]))
x = var()
run(0, x, conde([membero(x, (1, 2, 3))], [membero(x, (2, 3, 4))]))

# 调用lall包定义规则集合, lall(rules, [rules, ...])
x = var()
z = var()
rules = lall(
    eq(x, z),
    eq(z, 3)
)
run(0, x, rules)

