---
layout: post
title: 设计模式
date: 2024-10-27 10:00 +0800
tags: [设计模式]
toc: true
---

### 一、创建型模式

**1. 单例模式（Singleton Pattern）**

**概念：**
确保一个类只有一个实例，并提供一个全局访问点。

**应用场景：**
- 需要全局唯一的对象，例如日志记录器、线程池、配置管理器等。

**示例：**

```cpp
#include <iostream>
#include <string>

class Logger {
private:
    static Logger* instance;
    Logger() {} // 私有构造函数，防止外部实例化
public:
    static Logger* getInstance() {
        if (instance == nullptr) {
            instance = new Logger();
        }
        return instance;
    }
    void log(const std::string& message) {
        std::cout << "Log: " << message << std::endl;
    }
};

// 初始化静态成员
Logger* Logger::instance = nullptr;

// 使用示例
int main() {
    Logger* logger1 = Logger::getInstance();
    Logger* logger2 = Logger::getInstance();

    logger1->log("This is a log message.");

    if (logger1 == logger2) {
        std::cout << "Logger1 and Logger2 are the same instance." << std::endl;
    } else {
        std::cout << "Logger1 and Logger2 are different instances." << std::endl;
    }

    return 0;
}
```

**运行结果：**
```
Log: This is a log message.
Logger1 and Logger2 are the same instance.
```

---

**2. 工厂方法模式（Factory Method Pattern）**

**概念：**
定义一个创建对象的接口，但由子类决定要实例化的类是哪一个。

**应用场景：**
- 需要在不指定具体类的情况下创建对象。
- 客户端不需要知道所创建对象的类。

**示例：**

```cpp
#include <iostream>
#include <string>

class Product {
public:
    virtual void use() = 0;
};

class ConcreteProductA : public Product {
public:
    void use() override {
        std::cout << "Using Product A" << std::endl;
    }
};

class ConcreteProductB : public Product {
public:
    void use() override {
        std::cout << "Using Product B" << std::endl;
    }
};

class Factory {
public:
    virtual Product* createProduct() = 0;
};

class FactoryA : public Factory {
public:
    Product* createProduct() override {
        return new ConcreteProductA();
    }
};

class FactoryB : public Factory {
public:
    Product* createProduct() override {
        return new ConcreteProductB();
    }
};

// 使用示例
int main() {
    Factory* factory = new FactoryA();
    Product* product = factory->createProduct();
    product->use();

    delete product;
    delete factory;

    factory = new FactoryB();
    product = factory->createProduct();
    product->use();

    delete product;
    delete factory;

    return 0;
}
```

**运行结果：**
```
Using Product A
Using Product B
```

---

**3. 抽象工厂模式（Abstract Factory Pattern）**

**概念：**
提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类。

**应用场景：**
- 需要创建一组相关或相互依赖的对象。
- 系统需要独立于产品的创建方式。

**示例：**

```cpp
#include <iostream>
#include <string>

// 抽象产品：按钮
class Button {
public:
    virtual void draw() = 0;
};

// 具体产品：Windows风格按钮
class WinButton : public Button {
public:
    void draw() override {
        std::cout << "Draw a Windows Button." << std::endl;
    }
};

// 具体产品：Mac风格按钮
class MacButton : public Button {
public:
    void draw() override {
        std::cout << "Draw a Mac Button." << std::endl;
    }
};

// 抽象工厂
class GUIFactory {
public:
    virtual Button* createButton() = 0;
};

// 具体工厂：Windows工厂
class WinFactory : public GUIFactory {
public:
    Button* createButton() override {
        return new WinButton();
    }
};

// 具体工厂：Mac工厂
class MacFactory : public GUIFactory {
public:
    Button* createButton() override {
        return new MacButton();
    }
};

// 使用示例
int main() {
    GUIFactory* factory;
    Button* button;

    // 假设我们需要创建Windows风格的UI组件
    factory = new WinFactory();
    button = factory->createButton();
    button->draw();

    delete button;
    delete factory;

    // 创建Mac风格的UI组件
    factory = new MacFactory();
    button = factory->createButton();
    button->draw();

    delete button;
    delete factory;

    return 0;
}
```

**运行结果：**
```
Draw a Windows Button.
Draw a Mac Button.
```

---

**4. 建造者模式（Builder Pattern）**

**概念：**
将一个复杂对象的构建与其表示分离，使得同样的构建过程可以创建不同的表示。

**应用场景：**
- 需要创建复杂对象，其构建过程独立于部件的组成。
- 需要创建不同表示的对象。

**示例：**

```cpp
#include <iostream>
#include <string>

class Product {
public:
    void setPartA(const std::string& partA) { this->partA = partA; }
    void setPartB(const std::string& partB) { this->partB = partB; }
    void setPartC(const std::string& partC) { this->partC = partC; }
    void show() {
        std::cout << "Product Parts: " << partA << ", " << partB << ", " << partC << std::endl;
    }
private:
    std::string partA;
    std::string partB;
    std::string partC;
};

class Builder {
public:
    virtual void buildPartA() = 0;
    virtual void buildPartB() = 0;
    virtual void buildPartC() = 0;
    virtual Product* getResult() = 0;
};

class ConcreteBuilder : public Builder {
private:
    Product* product;
public:
    ConcreteBuilder() { product = new Product(); }
    ~ConcreteBuilder() { delete product; }
    void buildPartA() override { product->setPartA("Part A"); }
    void buildPartB() override { product->setPartB("Part B"); }
    void buildPartC() override { product->setPartC("Part C"); }
    Product* getResult() override { return product; }
};

class Director {
public:
    void construct(Builder* builder) {
        builder->buildPartA();
        builder->buildPartB();
        builder->buildPartC();
    }
};

// 使用示例
int main() {
    Director director;
    Builder* builder = new ConcreteBuilder();
    director.construct(builder);
    Product* product = builder->getResult();
    product->show();

    delete builder; // 注意，这里删除builder时也会删除product

    return 0;
}
```

**运行结果：**
```
Product Parts: Part A, Part B, Part C
```

---

### 二、结构型模式

**1. 适配器模式（Adapter Pattern）**

**概念：**
将一个类的接口转换成客户希望的另一个接口，使得原本由于接口不兼容而不能一起工作的类可以协同工作。

**应用场景：**
- 需要使用一个已有的类，而它的接口不符合需求。
- 希望创建一个可重用的类，用于与一些彼此之间没有太大关联的类一起工作。

**示例：**

```cpp
#include <iostream>
#include <string>

// 目标接口
class Target {
public:
    virtual void request() = 0;
};

// 需要适配的类
class Adaptee {
public:
    void specificRequest() {
        std::cout << "Adaptee's specific request." << std::endl;
    }
};

// 适配器
class Adapter : public Target {
private:
    Adaptee* adaptee;
public:
    Adapter(Adaptee* a) : adaptee(a) {}
    void request() override {
        adaptee->specificRequest();
    }
};

// 使用示例
int main() {
    Adaptee* adaptee = new Adaptee();
    Target* target = new Adapter(adaptee);
    target->request();

    delete target;
    delete adaptee;

    return 0;
}
```

**运行结果：**
```
Adaptee's specific request.
```

---

**2. 装饰器模式（Decorator Pattern）**

**概念：**
动态地给一个对象添加一些额外的职责，就增加功能来说，装饰器模式比生成子类更为灵活。

**应用场景：**
- 需要在不影响其他对象的情况下，以动态、透明的方式给单个对象添加职责。
- 需要对对象进行一些额外的功能拓展。

**示例：**

```cpp
#include <iostream>
#include <string>

class Component {
public:
    virtual void operation() = 0;
    virtual ~Component() {}
};

class ConcreteComponent : public Component {
public:
    void operation() override {
        std::cout << "ConcreteComponent operation." << std::endl;
    }
};

class Decorator : public Component {
protected:
    Component* component;
public:
    Decorator(Component* c) : component(c) {}
    virtual ~Decorator() { delete component; }
    void operation() override {
        component->operation();
    }
};

class ConcreteDecorator : public Decorator {
public:
    ConcreteDecorator(Component* c) : Decorator(c) {}
    void operation() override {
        Decorator::operation();
        addedBehavior();
    }
    void addedBehavior() {
        std::cout << "ConcreteDecorator added behavior." << std::endl;
    }
};

// 使用示例
int main() {
    Component* component = new ConcreteComponent();
    Component* decorator = new ConcreteDecorator(component);
    decorator->operation();

    delete decorator; // 会递归删除component

    return 0;
}
```

**运行结果：**
```
ConcreteComponent operation.
ConcreteDecorator added behavior.
```

---

**3. 代理模式（Proxy Pattern）**

**概念：**
为其他对象提供一种代理以控制对这个对象的访问。

**应用场景：**
- 需要为某对象提供代理以控制对它的访问。
- 远程代理、本地代理、虚代理等。

**示例：**

```cpp
#include <iostream>
#include <string>

class Subject {
public:
    virtual void request() = 0;
    virtual ~Subject() {}
};

class RealSubject : public Subject {
public:
    void request() override {
        std::cout << "RealSubject handling request." << std::endl;
    }
};

class Proxy : public Subject {
private:
    RealSubject* realSubject;
public:
    Proxy() : realSubject(nullptr) {}
    ~Proxy() { delete realSubject; }
    void request() override {
        if (realSubject == nullptr) {
            realSubject = new RealSubject();
        }
        std::cout << "Proxy delegating request to RealSubject." << std::endl;
        realSubject->request();
    }
};

// 使用示例
int main() {
    Subject* subject = new Proxy();
    subject->request();

    delete subject;

    return 0;
}
```

**运行结果：**
```
Proxy delegating request to RealSubject.
RealSubject handling request.
```

---

### 三、行为型模式

**1. 策略模式（Strategy Pattern）**

**概念：**
定义一系列的算法，把它们一个个封装起来，并且使它们可互相替换。

**应用场景：**
- 需要在不同的情况下使用不同的算法。
- 算法在运行时需要互相替换。

**示例：**

```cpp
#include <iostream>
#include <string>

class Strategy {
public:
    virtual void algorithmInterface() = 0;
    virtual ~Strategy() {}
};

class ConcreteStrategyA : public Strategy {
public:
    void algorithmInterface() override {
        std::cout << "Strategy A algorithm implementation." << std::endl;
    }
};

class ConcreteStrategyB : public Strategy {
public:
    void algorithmInterface() override {
        std::cout << "Strategy B algorithm implementation." << std::endl;
    }
};

class Context {
private:
    Strategy* strategy;
public:
    Context(Strategy* s) : strategy(s) {}
    ~Context() { delete strategy; }
    void contextInterface() {
        strategy->algorithmInterface();
    }
};

// 使用示例
int main() {
    Strategy* strategyA = new ConcreteStrategyA();
    Context* context = new Context(strategyA);
    context->contextInterface();

    delete context;

    Strategy* strategyB = new ConcreteStrategyB();
    context = new Context(strategyB);
    context->contextInterface();

    delete context;

    return 0;
}
```

**运行结果：**
```
Strategy A algorithm implementation.
Strategy B algorithm implementation.
```

---

**2. 观察者模式（Observer Pattern）**

**概念：**
定义对象间的一种一对多的依赖关系，当一个对象的状态发生改变时，所有依赖于它的对象都得到通知并被自动更新。

**应用场景：**
- 一个对象的改变需要同时改变其他对象，而且不知道有多少对象需要被改变。
- 一个抽象模型有两个方面，其中一个方面依赖于另一个方面。

**示例：**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

class Observer {
public:
    virtual void update() = 0;
    virtual ~Observer() {}
};

class ConcreteObserver : public Observer {
private:
    std::string name;
public:
    ConcreteObserver(const std::string& name) : name(name) {}
    void update() override {
        std::cout << "Observer " << name << " has been notified." << std::endl;
    }
};

class Subject {
private:
    std::vector<Observer*> observers;
public:
    void attach(Observer* o) {
        observers.push_back(o);
    }
    void detach(Observer* o) {
        observers.erase(std::remove(observers.begin(), observers.end(), o), observers.end());
    }
    void notify() {
        for (auto& observer : observers) {
            observer->update();
        }
    }
};

// 使用示例
int main() {
    Subject* subject = new Subject();

    Observer* observer1 = new ConcreteObserver("A");
    Observer* observer2 = new ConcreteObserver("B");
    Observer* observer3 = new ConcreteObserver("C");

    subject->attach(observer1);
    subject->attach(observer2);
    subject->attach(observer3);

    subject->notify();

    subject->detach(observer2);

    std::cout << "After detaching observer B:" << std::endl;
    subject->notify();

    delete observer1;
    delete observer2;
    delete observer3;
    delete subject;

    return 0;
}
```

**运行结果：**
```
Observer A has been notified.
Observer B has been notified.
Observer C has been notified.
After detaching observer B:
Observer A has been notified.
Observer C has been notified.
```

---

**3. 命令模式（Command Pattern）**

**概念：**
将一个请求封装为一个对象，从而使你可用不同的请求对客户进行参数化，对请求排队或记录请求日志，以及支持可撤销的操作。

**应用场景：**
- 需要对操作进行参数化。
- 需要将操作放入队列中执行、日志记录、支持撤销操作。

**示例：**

```cpp
#include <iostream>
#include <vector>

class Command {
public:
    virtual void execute() = 0;
    virtual ~Command() {}
};

class Receiver {
public:
    void action() {
        std::cout << "Receiver action performed." << std::endl;
    }
};

class ConcreteCommand : public Command {
private:
    Receiver* receiver;
public:
    ConcreteCommand(Receiver* r) : receiver(r) {}
    void execute() override {
        receiver->action();
    }
};

class Invoker {
private:
    std::vector<Command*> commands;
public:
    void setCommand(Command* c) {
        commands.push_back(c);
    }
    void executeCommands() {
        for (auto& command : commands) {
            command->execute();
        }
    }
    ~Invoker() {
        for (auto& command : commands) {
            delete command;
        }
    }
};

// 使用示例
int main() {
    Receiver* receiver = new Receiver();
    Command* command = new ConcreteCommand(receiver);

    Invoker* invoker = new Invoker();
    invoker->setCommand(command);
    invoker->executeCommands();

    delete invoker;
    delete receiver;

    return 0;
}
```

**运行结果：**
```
Receiver action performed.
```

---

**4. 状态模式（State Pattern）**

**概念：**
允许一个对象在其内部状态改变时改变它的行为，对象看起来似乎修改了它的类。

**应用场景：**
- 一个对象的行为取决于它的状态，并且它必须在运行时根据状态改变行为。

**示例：**

```cpp
#include <iostream>

class Context;

class State {
public:
    virtual void handle(Context* context) = 0;
    virtual ~State() {}
};

class ConcreteStateA : public State {
public:
    void handle(Context* context) override;
};

class ConcreteStateB : public State {
public:
    void handle(Context* context) override;
};

class Context {
private:
    State* state;
public:
    Context(State* s) : state(s) {}
    ~Context() { delete state; }
    void setState(State* s) {
        delete state;
        state = s;
    }
    void request() {
        state->handle(this);
    }
};

void ConcreteStateA::handle(Context* context) {
    std::cout << "State A handling request. Switching to State B." << std::endl;
    context->setState(new ConcreteStateB());
}

void ConcreteStateB::handle(Context* context) {
    std::cout << "State B handling request. Switching to State A." << std::endl;
    context->setState(new ConcreteStateA());
}

// 使用示例
int main() {
    Context* context = new Context(new ConcreteStateA());

    context->request();
    context->request();
    context->request();

    delete context;

    return 0;
}
```

**运行结果：**
```
State A handling request. Switching to State B.
State B handling request. Switching to State A.
State A handling request. Switching to State B.
```

---

**5. 模板方法模式（Template Method Pattern）**

**概念：**
定义一个操作中的算法的骨架，而将一些步骤延迟到子类中。模板方法使得子类可以不改变算法的结构即可重新定义算法的某些步骤。

**应用场景：**
- 一次性实现一个算法的不变部分，并将可变的行为留给子类来实现。
- 各子类中公共的行为被提取出来并集中到一个公共父类中，以避免代码重复。

**示例：**

```cpp
#include <iostream>

class AbstractClass {
public:
    void templateMethod() {
        primitiveOperation1();
        primitiveOperation2();
        hook();
    }
    virtual void primitiveOperation1() = 0;
    virtual void primitiveOperation2() = 0;
    virtual void hook() {} // 钩子方法，子类可选择性重写
    virtual ~AbstractClass() {}
};

class ConcreteClass : public AbstractClass {
public:
    void primitiveOperation1() override {
        std::cout << "ConcreteClass Operation1 implementation." << std::endl;
    }
    void primitiveOperation2() override {
        std::cout << "ConcreteClass Operation2 implementation." << std::endl;
    }
    void hook() override {
        std::cout << "ConcreteClass optional hook implementation." << std::endl;
    }
};

// 使用示例
int main() {
    AbstractClass* obj = new ConcreteClass();
    obj->templateMethod();

    delete obj;

    return 0;
}
```

**运行结果：**
```
ConcreteClass Operation1 implementation.
ConcreteClass Operation2 implementation.
ConcreteClass optional hook implementation.
```

---

### 总结

通过在每个设计模式的示例中添加`main`函数，我们可以更直观地看到这些模式在实际开发中的应用方式。每个`main`函数都演示了如何创建和使用相应的设计模式，帮助理解其工作原理和使用场景。在实际项目中，根据具体需求和场景，选择合适的设计模式能够提高代码的可维护性、可扩展性和可读性。