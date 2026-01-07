"""
依赖注入容器

提供简单的依赖注入容器，管理应用程序的服务实例。
"""

import logging
from typing import Dict, Any, Callable, Optional, TypeVar, Type
from threading import Lock
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Scope(str, Enum):
    """依赖作用域"""
    SINGLETON = "singleton"  # 单例，整个应用共享
    TRANSIENT = "transient"  # 瞬态，每次获取都创建新实例
    SCOPED = "scoped"        # 作用域，在同一作用域内共享


@dataclass
class ServiceDescriptor:
    """服务描述符"""
    key: str
    factory: Callable[..., Any]
    scope: Scope = Scope.SINGLETON
    instance: Optional[Any] = None


class Container:
    """依赖注入容器

    支持单例、瞬态和作用域服务的注册和解析。

    Usage:
        container = Container()

        # 注册单例服务
        container.register_singleton('database', lambda: Database())

        # 注册工厂函数
        container.register('cache', lambda c: APICache(c.get('config').cache_size))

        # 获取服务
        db = container.get('database')

        # 使用类型提示
        db: Database = container.get('database', Database)
    """

    def __init__(self):
        self._services: Dict[str, ServiceDescriptor] = {}
        self._lock = Lock()
        self._scoped_instances: Dict[str, Dict[str, Any]] = {}
        self._current_scope: Optional[str] = None

    def register(
        self,
        key: str,
        factory: Callable[..., T],
        scope: Scope = Scope.SINGLETON
    ) -> 'Container':
        """注册服务

        Args:
            key: 服务标识符
            factory: 工厂函数，接收容器作为参数
            scope: 服务作用域

        Returns:
            容器实例（支持链式调用）
        """
        with self._lock:
            self._services[key] = ServiceDescriptor(
                key=key,
                factory=factory,
                scope=scope
            )
            logger.debug(f"注册服务: {key} (scope={scope.value})")
        return self

    def register_singleton(self, key: str, factory: Callable[..., T]) -> 'Container':
        """注册单例服务"""
        return self.register(key, factory, Scope.SINGLETON)

    def register_transient(self, key: str, factory: Callable[..., T]) -> 'Container':
        """注册瞬态服务"""
        return self.register(key, factory, Scope.TRANSIENT)

    def register_instance(self, key: str, instance: T) -> 'Container':
        """直接注册实例"""
        with self._lock:
            self._services[key] = ServiceDescriptor(
                key=key,
                factory=lambda: instance,
                scope=Scope.SINGLETON,
                instance=instance
            )
            logger.debug(f"注册实例: {key}")
        return self

    def get(self, key: str, expected_type: Optional[Type[T]] = None) -> T:
        """获取服务实例

        Args:
            key: 服务标识符
            expected_type: 期望的类型（用于类型提示）

        Returns:
            服务实例

        Raises:
            KeyError: 服务未注册
        """
        with self._lock:
            if key not in self._services:
                raise KeyError(f"服务未注册: {key}")

            descriptor = self._services[key]

            # 单例：已有实例则直接返回
            if descriptor.scope == Scope.SINGLETON:
                if descriptor.instance is None:
                    descriptor.instance = self._create_instance(descriptor)
                return descriptor.instance

            # 瞬态：每次创建新实例
            if descriptor.scope == Scope.TRANSIENT:
                return self._create_instance(descriptor)

            # 作用域：在当前作用域内共享
            if descriptor.scope == Scope.SCOPED:
                if self._current_scope is None:
                    raise RuntimeError("作用域服务必须在作用域内使用")
                scope_instances = self._scoped_instances.get(self._current_scope, {})
                if key not in scope_instances:
                    scope_instances[key] = self._create_instance(descriptor)
                    self._scoped_instances[self._current_scope] = scope_instances
                return scope_instances[key]

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        try:
            # 工厂函数可以接收容器作为参数
            import inspect
            sig = inspect.signature(descriptor.factory)
            if sig.parameters:
                return descriptor.factory(self)
            else:
                return descriptor.factory()
        except Exception as e:
            logger.error(f"创建服务失败 [{descriptor.key}]: {e}")
            raise

    def has(self, key: str) -> bool:
        """检查服务是否已注册"""
        return key in self._services

    def remove(self, key: str) -> bool:
        """移除服务"""
        with self._lock:
            if key in self._services:
                del self._services[key]
                return True
            return False

    def reset(self, key: Optional[str] = None):
        """重置服务实例

        Args:
            key: 指定服务，为 None 则重置所有
        """
        with self._lock:
            if key:
                if key in self._services:
                    self._services[key].instance = None
            else:
                for descriptor in self._services.values():
                    descriptor.instance = None
                self._scoped_instances.clear()
            logger.debug(f"重置服务: {key or 'all'}")

    def enter_scope(self, scope_id: str):
        """进入作用域"""
        self._current_scope = scope_id
        self._scoped_instances[scope_id] = {}

    def exit_scope(self, scope_id: str):
        """退出作用域"""
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
        if self._current_scope == scope_id:
            self._current_scope = None

    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """列出所有注册的服务"""
        return {
            key: {
                "scope": descriptor.scope.value,
                "has_instance": descriptor.instance is not None
            }
            for key, descriptor in self._services.items()
        }


# ==================== 全局容器 ====================

_container: Optional[Container] = None


def get_container() -> Container:
    """获取全局容器实例"""
    global _container
    if _container is None:
        _container = Container()
    return _container


def configure_container(container: Container):
    """配置全局容器"""
    global _container
    _container = container


def reset_container():
    """重置全局容器（用于测试）"""
    global _container
    _container = None


# ==================== 便捷装饰器 ====================

def inject(*dependencies: str):
    """依赖注入装饰器

    Usage:
        @inject('database', 'cache')
        def my_function(db, cache, other_arg):
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            container = get_container()
            injected = [container.get(dep) for dep in dependencies]
            return func(*injected, *args, **kwargs)
        return wrapper
    return decorator


# ==================== 应用初始化 ====================

def setup_default_services(container: Optional[Container] = None):
    """设置默认服务

    初始化应用程序所需的所有服务。
    """
    c = container or get_container()

    # 数据库
    c.register_singleton('database', lambda: _lazy_import('database', 'Database')())

    # 缓存
    c.register_singleton('api_cache', lambda: _lazy_import('cache', 'APICache')(
        maxsize=1000, ttl=300
    ))
    c.register_singleton('session_cache', lambda: _lazy_import('cache', 'SessionCache')(
        maxsize=10000, ttl=1800
    ))

    # 意图注册中心
    c.register_singleton('intent_registry', lambda: _lazy_import('intent_registry', 'IntentRegistry')())

    # 向量存储
    c.register_singleton('retriever', lambda: _lazy_import('vector_store', 'get_retriever')())

    # 重试管理器
    c.register_singleton('retry_manager', lambda: _lazy_import('retry_manager', 'RetryManager')(
        policy=_lazy_import('retry_manager', 'ExponentialBackoffPolicy')(),
        max_attempts=3
    ))

    # 健康检查器
    c.register_singleton('health_checker', lambda: _lazy_import('health', 'get_health_checker')())

    logger.info(f"已注册 {len(c.list_services())} 个默认服务")
    return c


def _lazy_import(module_name: str, attr_name: str):
    """延迟导入模块属性"""
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
