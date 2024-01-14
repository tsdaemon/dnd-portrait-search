from pydantic import BaseModel


class BaseEntity(BaseModel):
    def validate_entity(self) -> None:
        _ = self.__class__.model_validate(self.__dict__)
