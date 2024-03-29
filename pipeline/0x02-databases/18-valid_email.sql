-- Creates a trigger that resets the attribute valid_email only when the email has been changed.
delimiter //
CREATE TRIGGER update_email
	BEFORE UPDATE
	ON users
	FOR EACH ROW
	BEGIN
  		IF STRCMP(OLD.email, NEW.email) <> 0 THEN
			SET NEW.valid_email = 0;
		END IF;
	END //
delimiter ;
