class PremError(Exception):
    pass


class PremProviderError(PremError):
    def __init__(self, message, provider, model, provider_message):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.provider_message = provider_message
        self.model = model


class PremProviderNotFoundError(PremProviderError):
    pass


class PremProviderAPIErrror(PremProviderError):
    pass


class PremProviderAuthenticationError(PremProviderError):
    pass


class PremProviderConflictError(PremProviderError):
    pass


class PremProviderAPIStatusError(PremProviderError):
    pass


class PremProviderAPITimeoutError(PremProviderError):
    pass


class PremProviderRateLimitError(PremProviderError):
    pass


class PremProviderBadRequestError(PremProviderError):
    pass


class PremProviderAPIConnectionError(PremProviderError):
    pass


class PremProviderInternalServerError(PremProviderError):
    pass


class PremProviderPermissionDeniedError(PremProviderError):
    pass


class PremProviderUnprocessableEntityError(PremProviderError):
    pass


class PremProviderAPIResponseValidationError(PremProviderError):
    pass


class PremProviderResponseValidationError(PremProviderError):
    pass
