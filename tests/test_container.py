"""
依赖注入容器测试
"""

import pytest

import sys
sys.path.insert(0, '/Users/sawzhang/code/AIChatOrder')

from container import (
    Container, Scope,
    get_container, reset_container, configure_container
)


class TestContainer:
    """Container 测试"""

    @pytest.fixture
    def container(self):
        """创建测试用容器"""
        return Container()

    def test_register_and_get_singleton(self, container):
        """测试注册和获取单例"""
        container.register_singleton('service', lambda: {"id": 1})

        service1 = container.get('service')
        service2 = container.get('service')

        assert service1 is service2
        assert service1["id"] == 1

    def test_register_transient(self, container):
        """测试瞬态服务"""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        container.register_transient('service', factory)

        service1 = container.get('service')
        service2 = container.get('service')

        assert service1 is not service2
        assert service1["count"] == 1
        assert service2["count"] == 2

    def test_register_instance(self, container):
        """测试直接注册实例"""
        instance = {"key": "value"}
        container.register_instance('service', instance)

        result = container.get('service')

        assert result is instance

    def test_get_not_found(self, container):
        """测试获取不存在的服务"""
        with pytest.raises(KeyError) as exc_info:
            container.get('nonexistent')

        assert "服务未注册" in str(exc_info.value)

    def test_has(self, container):
        """测试检查服务是否存在"""
        container.register_singleton('service', lambda: {})

        assert container.has('service') is True
        assert container.has('nonexistent') is False

    def test_remove(self, container):
        """测试移除服务"""
        container.register_singleton('service', lambda: {})

        assert container.remove('service') is True
        assert container.has('service') is False
        assert container.remove('service') is False

    def test_reset_single(self, container):
        """测试重置单个服务"""
        container.register_singleton('service', lambda: {"id": 1})
        container.get('service')  # 创建实例

        container.reset('service')

        # 应该重新创建实例
        assert container._services['service'].instance is None

    def test_reset_all(self, container):
        """测试重置所有服务"""
        container.register_singleton('s1', lambda: {})
        container.register_singleton('s2', lambda: {})
        container.get('s1')
        container.get('s2')

        container.reset()

        assert container._services['s1'].instance is None
        assert container._services['s2'].instance is None

    def test_factory_with_container(self, container):
        """测试工厂函数接收容器参数"""
        container.register_singleton('config', lambda: {"db_url": "sqlite:///"})
        container.register_singleton('database', lambda c: {
            "url": c.get('config')["db_url"]
        })

        db = container.get('database')

        assert db["url"] == "sqlite:///"

    def test_chain_registration(self, container):
        """测试链式注册"""
        result = (container
            .register_singleton('s1', lambda: {})
            .register_singleton('s2', lambda: {})
            .register_instance('s3', {}))

        assert result is container
        assert container.has('s1')
        assert container.has('s2')
        assert container.has('s3')

    def test_list_services(self, container):
        """测试列出服务"""
        container.register_singleton('s1', lambda: {})
        container.register_transient('s2', lambda: {})
        container.get('s1')

        services = container.list_services()

        assert 's1' in services
        assert 's2' in services
        assert services['s1']['scope'] == 'singleton'
        assert services['s1']['has_instance'] is True
        assert services['s2']['scope'] == 'transient'
        assert services['s2']['has_instance'] is False


class TestScoped:
    """作用域服务测试"""

    def test_scoped_service(self):
        """测试作用域服务"""
        container = Container()
        container.register('service', lambda: {"id": id({})}, Scope.SCOPED)

        container.enter_scope('request1')
        s1a = container.get('service')
        s1b = container.get('service')

        container.exit_scope('request1')
        container.enter_scope('request2')
        s2 = container.get('service')
        container.exit_scope('request2')

        assert s1a is s1b  # 同一作用域内相同
        assert s1a is not s2  # 不同作用域不同

    def test_scoped_without_scope(self):
        """测试不在作用域内获取作用域服务"""
        container = Container()
        container.register('service', lambda: {}, Scope.SCOPED)

        with pytest.raises(RuntimeError) as exc_info:
            container.get('service')

        assert "作用域服务必须在作用域内使用" in str(exc_info.value)


class TestGlobalContainer:
    """全局容器测试"""

    def test_get_container_singleton(self):
        """测试全局容器单例"""
        reset_container()
        c1 = get_container()
        c2 = get_container()

        assert c1 is c2

    def test_configure_container(self):
        """测试配置全局容器"""
        reset_container()
        custom = Container()
        custom.register_instance('custom', {"custom": True})

        configure_container(custom)

        assert get_container() is custom
        assert get_container().get('custom')["custom"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
