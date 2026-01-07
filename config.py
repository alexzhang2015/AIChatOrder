"""
配置管理系统

使用 Pydantic Settings 管理应用配置，支持环境变量和配置文件。
"""

import os
import logging
from typing import List, Optional
from pathlib import Path
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class OpenAISettings(BaseSettings):
    """OpenAI 相关配置"""
    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    base_url: Optional[str] = Field(default=None, description="OpenAI API Base URL")
    model: str = Field(default="gpt-4o-mini", description="使用的模型")
    timeout: float = Field(default=30.0, ge=5.0, le=120.0, description="请求超时时间")
    max_retries: int = Field(default=3, ge=1, le=10, description="最大重试次数")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="生成温度")
    max_tokens: int = Field(default=500, ge=100, le=4000, description="最大生成 token 数")


class DatabaseSettings(BaseSettings):
    """数据库相关配置"""
    model_config = SettingsConfigDict(env_prefix="DB_")

    path: Path = Field(
        default=Path("data/coffee_order.db"),
        description="数据库文件路径"
    )
    pool_size: int = Field(default=10, ge=1, le=50, description="连接池大小")
    timeout: float = Field(default=30.0, ge=5.0, le=120.0, description="连接超时时间")
    wal_mode: bool = Field(default=True, description="是否启用 WAL 模式")


class CacheSettings(BaseSettings):
    """缓存相关配置"""
    model_config = SettingsConfigDict(env_prefix="CACHE_")

    api_maxsize: int = Field(default=1000, ge=100, le=10000, description="API 缓存最大条目数")
    api_ttl: int = Field(default=300, ge=60, le=3600, description="API 缓存过期时间(秒)")
    session_maxsize: int = Field(default=10000, ge=100, le=100000, description="会话缓存最大数")
    session_ttl: int = Field(default=1800, ge=60, le=7200, description="会话缓存过期时间(秒)")
    vector_enabled: bool = Field(default=True, description="是否启用向量检索缓存")
    vector_ttl: int = Field(default=600, ge=60, le=3600, description="向量检索缓存过期时间(秒)")


class RateLimitSettings(BaseSettings):
    """限流相关配置"""
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")

    enabled: bool = Field(default=True, description="是否启用限流")
    requests_per_second: float = Field(default=10.0, ge=0.1, le=1000.0, description="每秒请求数")
    burst_size: int = Field(default=20, ge=1, le=100, description="突发容量")
    per_ip: bool = Field(default=True, description="是否按 IP 限流")


class CircuitBreakerSettings(BaseSettings):
    """熔断器相关配置"""
    model_config = SettingsConfigDict(env_prefix="CIRCUIT_BREAKER_")

    enabled: bool = Field(default=True, description="是否启用熔断器")
    failure_threshold: int = Field(default=5, ge=1, le=100, description="失败阈值")
    success_threshold: int = Field(default=3, ge=1, le=50, description="恢复阈值")
    timeout: float = Field(default=30.0, ge=5.0, le=300.0, description="熔断超时时间(秒)")


class ServerSettings(BaseSettings):
    """服务器相关配置"""
    model_config = SettingsConfigDict(env_prefix="SERVER_")

    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, ge=1, le=65535, description="监听端口")
    debug: bool = Field(default=False, description="调试模式")
    workers: int = Field(default=1, ge=1, le=16, description="工作进程数")


class LoggingSettings(BaseSettings):
    """日志相关配置"""
    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(default="structured", description="日志格式 (structured/plain)")
    file: Optional[Path] = Field(default=None, description="日志文件路径")

    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"无效的日志级别: {v}, 有效值: {valid_levels}")
        return v


class CORSSettings(BaseSettings):
    """CORS 相关配置"""
    model_config = SettingsConfigDict(env_prefix="CORS_")

    origins: List[str] = Field(
        default=["*"],
        description="允许的来源"
    )
    allow_credentials: bool = Field(default=True, description="是否允许凭证")
    allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="允许的方法"
    )
    max_age: int = Field(default=600, ge=0, le=86400, description="预检请求缓存时间(秒)")


class Settings(BaseSettings):
    """应用主配置"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 应用信息
    app_name: str = Field(default="AI 点单意图识别系统", description="应用名称")
    app_version: str = Field(default="3.2.0", description="应用版本")
    environment: str = Field(default="development", description="运行环境")

    # 子配置
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = ['development', 'staging', 'production', 'testing']
        v = v.lower()
        if v not in valid_envs:
            raise ValueError(f"无效的环境: {v}, 有效值: {valid_envs}")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    def to_dict(self) -> dict:
        """转换为字典（隐藏敏感信息）"""
        data = {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
            "openai": {
                "model": self.openai.model,
                "base_url": self.openai.base_url,
                "has_api_key": bool(self.openai.api_key),
                "timeout": self.openai.timeout,
                "max_retries": self.openai.max_retries
            },
            "database": {
                "path": str(self.database.path),
                "pool_size": self.database.pool_size,
                "wal_mode": self.database.wal_mode
            },
            "cache": {
                "api_maxsize": self.cache.api_maxsize,
                "api_ttl": self.cache.api_ttl,
                "session_maxsize": self.cache.session_maxsize,
                "session_ttl": self.cache.session_ttl,
                "vector_enabled": self.cache.vector_enabled
            },
            "rate_limit": {
                "enabled": self.rate_limit.enabled,
                "requests_per_second": self.rate_limit.requests_per_second,
                "burst_size": self.rate_limit.burst_size
            },
            "circuit_breaker": {
                "enabled": self.circuit_breaker.enabled,
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "timeout": self.circuit_breaker.timeout
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format
            }
        }
        return data


# ==================== 全局实例 ====================

_settings: Optional[Settings] = None


@lru_cache()
def get_settings() -> Settings:
    """获取配置实例（缓存）"""
    global _settings
    if _settings is None:
        _settings = Settings()
        logger.info(f"配置已加载: {_settings.environment} 环境")
    return _settings


def reload_settings() -> Settings:
    """重新加载配置"""
    global _settings
    get_settings.cache_clear()
    _settings = None
    return get_settings()


def get_openai_settings() -> OpenAISettings:
    """获取 OpenAI 配置"""
    return get_settings().openai


def get_database_settings() -> DatabaseSettings:
    """获取数据库配置"""
    return get_settings().database


def get_cache_settings() -> CacheSettings:
    """获取缓存配置"""
    return get_settings().cache
