from DeepQ import DeepQ
from MathLangEnv import MathLangDeepQEnvAdapter

if __name__ == '__main__':
    trainer = DeepQ("police net :)")
    trainer.train(1, MathLangDeepQEnvAdapter())
    print("Deep Q Main Program Finished Execution")