import numpy as np

class MultipleLogisticRegressioniClassifier:
    
    def __init__(self):
        """
        init_state: 重みの初期化の有無
        W: 重み
        b: 閾値
        loss: 損失
        val_loss: テストでのロス
        acc: 正答率
        val_acc: テストでの正答率
        """
        self.init_state = True
        self.W = None
        self.b = None
        self.W_hist = None
        self.b_hist = None
        self.loss = np.array([])
        self.val_loss = np.array([])
        self.acc = np.array([])
        self.val_acc = np.array([])

    def fit(self, X, y, X_test, y_test, steps=100, lr=0.01, verbose=True, history=False):
        """
        # 学習
        ------------------------
        X: 学習データ
        y: ラベル
        X_test: テストデータ
        y_test: テスト用ラベル
        steps: 重みの更新回数
        lr: 学習率
        verbose: 学習経過を表示
        history: 重みの更新履歴
        ------------------------
        """

        # DataFrameをarrayに変換
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # サンプルサイズ，説明変数の数，クラスの数
        N, n_features = X.shape
        n_classes = y.shape[1]

        # 重みの初期化
        if self.init_state:
            self.W = np.random.normal(0, 1/np.sqrt(N), [n_features, n_classes])
            self.b = np.random.normal(0, 1/np.sqrt(N), [1, n_classes])
            self.init_state = False

        # 重みの更新履歴の初期化
        if history:
            self.W_hist = np.empty((0, n_features, n_classes), float)
            self.b_hist = np.empty((0, 1, n_classes), float)

        

        for step in range(steps):
            # 学習
            self.loss = np.append(self.loss, self.__cross_entropy(X,y))
            self.val_loss = np.append(self.val_loss, self.__cross_entropy(X_test,y_test))
            self.acc = np.append(self.acc, self.__accuracy(y, self.predict(X)))
            self.val_acc = np.append(self.val_acc, self.__accuracy(y_test, self.predict(X_test)))

            # 重みの更新
            gW, gb = self.__grad(X, y)
            self.W -= lr * gW
            self.b -= lr * gb

            # 重みを記録
            self.W_hist = np.append(self.W_hist, self.W[np.newaxis], axis=0)
            self.b_hist = np.append(self.b_hist, self.b[np.newaxis], axis=0)

            if verbose != False:
                print("Step " + str(step + 1) + "/" + str(steps) + \
                      "    loss: " + str(self.loss[-1]) + \
                      "    acc: " + str(self.acc[-1]) + \
                      "    val_loss: " + str(self.val_loss[-1]) + \
                      "    val_acc: " + str(self.val_acc[-1]))



    def predict(self, X):
        pred = self.__prediction(X)
        _y = np.zeros([X.shape[0], self.W.shape[1]])

        class_label = np.argmax(pred, axis=1)

        for i in range(len(_y)):
            _y[i, class_label[i]] = 1

        return _y

    def __softmax(self, z):
        """
        ソフトマックス関数
        """
        return np.exp(z) / np.sum(np.exp(z), axis=1)[:, np.newaxis]

    def __cross_entropy(self, X, y, e=1e-12):
        """
        交差エントロピー
        """
        pred = self.__prediction(X)
        X = np.clip(X, e, 1.-e)
        N = X.shape[0]
        return -np.sum(y * np.log(pred))/N

    def __grad(self, X, y):
        """
        損失の勾配
        """
        N = X.shape[0]
        pred = self.__prediction(X)
        gW = -np.dot(X.T, y-pred) / N
        gb = -np.dot(np.ones([1, N]), y-pred) / N
        return gW, gb

    def __prediction(self, X):
        """
        予測値
        """
        return self.__softmax(np.dot(X, self.W) + self.b)

    def __accuracy(self, y, _y):
        """
        正答率
        y: 真のクラス
        _y: 予測クラス
        """
        acc = np.array([np.sum(y[i, :]==_y[i, :])==len(y[i, :]) for i in range(len(y))])
        return np.sum(acc) / len(acc)


